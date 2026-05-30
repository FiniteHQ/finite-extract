import Foundation

// MARK: - Generation

/// Runs the language model: builds a user prompt from the context, generates text,
/// and stores it on `context.intermediate`. The system prompt and prompt-builder are
/// caller-supplied, so this single stage serves both the simple JSON path and a
/// natural-language-intermediate path.
public struct GenerateStage: ExtractionStage {
    public let name: String
    private let extractor: FiniteExtract
    private let systemPrompt: String
    private let promptBuilder: @Sendable (ExtractionContext) -> String
    private let maxTokens: Int
    private let temperature: Float

    /// Full form: the prompt builder sees the whole context (e.g. to fold in `typedSpans`).
    public init(
        _ extractor: FiniteExtract,
        systemPrompt: String,
        name: String = "generate",
        maxTokens: Int = 2048,
        temperature: Float = 0.0,
        prompt promptBuilder: @escaping @Sendable (ExtractionContext) -> String
    ) {
        self.extractor = extractor
        self.systemPrompt = systemPrompt
        self.name = name
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.promptBuilder = promptBuilder
    }

    /// Common form: the prompt is built from the input text alone.
    public init(
        _ extractor: FiniteExtract,
        systemPrompt: String,
        name: String = "generate",
        maxTokens: Int = 2048,
        temperature: Float = 0.0,
        userPrompt: @escaping @Sendable (String) -> String
    ) {
        self.init(
            extractor, systemPrompt: systemPrompt, name: name,
            maxTokens: maxTokens, temperature: temperature
        ) { context in userPrompt(context.inputText) }
    }

    public func run(_ context: inout ExtractionContext) async throws {
        let userPrompt = promptBuilder(context)
        let (text, _) = try await extractor.generateText(
            systemPrompt: systemPrompt,
            userPrompt: userPrompt,
            maxTokens: maxTokens,
            temperature: temperature
        )
        context.intermediate = text
        context.userInfo["modelName"] = extractor.modelName
    }
}

// MARK: - Assembly (default: JSON repair)

/// The default assemble stage for the simple path: repairs the model's JSON output
/// (`Postprocessor.extractJSON`), applies an optional caller postprocess closure, and
/// stores the result on `context.json`. Pipelines that emit a non-JSON intermediate
/// (e.g. a natural-language format) replace this with a custom assemble stage.
public struct JSONRepairStage: ExtractionStage {
    public let name = "json-repair"
    private let postprocess: (@Sendable (inout [String: Any], String) -> Void)?

    public init(postprocess: (@Sendable (inout [String: Any], String) -> Void)? = nil) {
        self.postprocess = postprocess
    }

    public func run(_ context: inout ExtractionContext) async throws {
        guard let raw = context.intermediate else {
            throw ExtractionError.invalidJSON(rawOutput: "")
        }
        let (_, parsed) = try FiniteExtract.repairAndPostprocess(
            raw, postprocess: postprocess, inputText: context.inputText
        )
        context.json = parsed
    }
}

// MARK: - Typing pre-pass

/// A discriminative entity typer. Given input text, returns typed spans. Implementations
/// range from a pure lexicon lookup to a Core ML NER model — the engine ships both the
/// protocol and reference implementations; callers can supply their own.
public protocol EntityTyper: Sendable {
    func typedSpans(in text: String) async throws -> [TypedSpan]
}

/// Wraps an `EntityTyper` as a stage: fills `context.typedSpans` so later stages
/// (a generate stage that annotates the note, or an assemble stage that filters the
/// model's output) can rely on authoritative typing.
public struct TypingStage: ExtractionStage {
    public let name: String
    private let typer: any EntityTyper

    public init(_ typer: any EntityTyper, name: String = "typing") {
        self.typer = typer
        self.name = name
    }

    public func run(_ context: inout ExtractionContext) async throws {
        context.typedSpans = try await typer.typedSpans(in: context.inputText)
    }
}

// MARK: - Reference typers

/// Collision-aware lexicon typer. The caller supplies, per label, the set of names that
/// map to it. A name appearing under more than one label is ambiguous and is dropped
/// (the typer abstains, deferring to a fallback such as an NER model). This is the
/// deterministic fast path described in the research (§17.1): the names that are
/// lexically unambiguous are typed for free, with no model call.
public struct LexiconEntityTyper: EntityTyper {
    /// name(lowercased) -> label, only for names unambiguous across labels.
    private let table: [String: String]

    /// - Parameter lexicons: label -> the set of names for that label.
    public init(_ lexicons: [String: Set<String>]) {
        var counts: [String: Int] = [:]
        var assignment: [String: String] = [:]
        for (label, names) in lexicons {
            for raw in names {
                let key = raw.lowercased()
                counts[key, default: 0] += 1
                assignment[key] = label
            }
        }
        self.table = assignment.filter { counts[$0.key] == 1 }
    }

    public func typedSpans(in text: String) -> [TypedSpan] {
        var spans: [TypedSpan] = []
        var seen = Set<String>()
        for candidate in Self.candidateNames(in: text) {
            let key = candidate.text.lowercased()
            guard !seen.contains(key), let label = table[key] else { continue }
            seen.insert(key)
            spans.append(TypedSpan(
                text: candidate.text, label: label, score: 1.0,
                start: candidate.start, end: candidate.end
            ))
        }
        return spans
    }

    // swiftlint:disable large_tuple
    // The (text, start, end) span triple is a lightweight internal return type. If it
    // gains more fields it should become a named struct, but three is fine here.
    /// Capitalised single-token candidate names. The reference lexicon is keyed on
    /// single-token names (first names, pet names), so unigram extraction is both
    /// correct and avoids greedily fusing adjacent capitals ("Met Sarah"). Multi-token
    /// names are the job of an NER fallback, not the deterministic fast path.
    static func candidateNames(in text: String) -> [(text: String, start: Int, end: Int)] {
        let pattern = "\\b[A-Z][a-zA-Z]+\\b"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        return matches.map { m in
            (text: ns.substring(with: m.range), start: m.range.location, end: m.range.location + m.range.length)
        }
    }
    // swiftlint:enable large_tuple
}

/// Two-stage typer mirroring the research pipeline (§17.1): a deterministic lexicon
/// fast path, then a `fallback` typer (e.g. a Core ML NER model) for the names the
/// lexicon abstained on. Lexicon decisions win; the fallback only sees the remainder.
public struct CompositeEntityTyper: EntityTyper {
    private let lexicon: LexiconEntityTyper
    private let fallback: any EntityTyper

    public init(lexicon: LexiconEntityTyper, fallback: any EntityTyper) {
        self.lexicon = lexicon
        self.fallback = fallback
    }

    public func typedSpans(in text: String) async throws -> [TypedSpan] {
        let lex = lexicon.typedSpans(in: text)
        let decided = Set(lex.map { $0.text.lowercased() })
        let extra = try await fallback.typedSpans(in: text)
            .filter { !decided.contains($0.text.lowercased()) }
        return lex + extra
    }
}

import Foundation

/// A typed entity span produced by a discriminative pre-pass (e.g. an NER `EntityTyper`).
///
/// The `label` vocabulary belongs to the caller — the engine never assumes a fixed
/// set like "person"/"pet". A downstream assemble stage decides how to use the typing.
public struct TypedSpan: Sendable, Equatable {
    /// The surface text of the entity as it appears in the input.
    public let text: String
    /// The caller-defined type label (e.g. "person", "pet", "org").
    public let label: String
    /// Confidence in `[0, 1]`. Deterministic sources (a lexicon hit) use `1.0`.
    public let score: Double
    /// UTF-16 character offsets into the input, if the typer reports them.
    public let start: Int?
    public let end: Int?

    public init(text: String, label: String, score: Double = 1.0, start: Int? = nil, end: Int? = nil) {
        self.text = text
        self.label = label
        self.score = score
        self.start = start
        self.end = end
    }
}

/// Timing record for one stage, surfaced on `ExtractionMetadata.stages`.
public struct StageTrace: Sendable, Equatable {
    public let name: String
    public let milliseconds: Int
    public init(name: String, milliseconds: Int) {
        self.name = name
        self.milliseconds = milliseconds
    }
}

/// Mutable state threaded through an `ExtractionPipeline`. The input is immutable;
/// stages populate the accumulators in order. A typical flow:
/// a typing stage fills `typedSpans`, a generate stage fills `intermediate`, and an
/// assemble stage reads both and fills `json`.
///
/// `@unchecked Sendable` mirrors `ExtractionResult`: the `[String: Any]` members hold
/// Foundation value types and the context is only ever mutated sequentially within a
/// single pipeline run, never shared across concurrency boundaries.
public struct ExtractionContext: @unchecked Sendable {
    /// The original input text. Immutable for the run.
    public let inputText: String
    /// Model output (NL / DSL / JSON), set by a generation stage.
    public var intermediate: String?
    /// Typed entity spans from a discriminative pre-pass. Empty when no typing stage runs.
    public var typedSpans: [TypedSpan]
    /// The evolving structured result. An assemble stage is responsible for filling this.
    public var json: [String: Any]
    /// Free-form scratch space so custom stages can pass data to later stages.
    public var userInfo: [String: Any]
    /// Per-stage timings, appended by the pipeline as each stage completes.
    public internal(set) var trace: [StageTrace]

    public init(
        inputText: String,
        intermediate: String? = nil,
        typedSpans: [TypedSpan] = [],
        json: [String: Any] = [:],
        userInfo: [String: Any] = [:]
    ) {
        self.inputText = inputText
        self.intermediate = intermediate
        self.typedSpans = typedSpans
        self.json = json
        self.userInfo = userInfo
        self.trace = []
    }
}

/// One step in an extraction pipeline. A stage reads from and writes to the shared
/// `ExtractionContext`. Stages run in declaration order.
public protocol ExtractionStage: Sendable {
    /// Short identifier surfaced in `StageTrace`.
    var name: String { get }
    /// Transform the context. Throwing aborts the pipeline.
    func run(_ context: inout ExtractionContext) async throws
}

/// An ordered composition of `ExtractionStage`s.
///
/// The single-pass path is just two stages — generation then JSON repair — and is what
/// `FiniteExtract.extract(from:schema:)` builds internally. Richer pipelines insert a
/// typing pre-pass and swap the repair stage for a deterministic assembler:
///
/// ```swift
/// let pipeline = ExtractionPipeline([
///     TypingStage(myTyper),                       // fills context.typedSpans
///     GenerateStage(extractor, systemPrompt: s) { $0 },
///     MyAssembleStage(),                          // reads intermediate + typedSpans → json
/// ])
/// let result = try await pipeline.run(on: text)
/// ```
public struct ExtractionPipeline: Sendable {
    public let stages: [any ExtractionStage]

    public init(_ stages: [any ExtractionStage]) {
        self.stages = stages
    }

    /// Run every stage in order and assemble an `ExtractionResult` from the final context.
    @discardableResult
    public func run(on text: String) async throws -> ExtractionResult {
        var context = ExtractionContext(inputText: text)

        for stage in stages {
            try Task.checkCancellation()
            let start = CFAbsoluteTimeGetCurrent()
            try await stage.run(&context)
            let ms = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)
            context.trace.append(StageTrace(name: stage.name, milliseconds: ms))
        }

        let parsed = context.json
        guard JSONSerialization.isValidJSONObject(parsed) else {
            throw ExtractionError.invalidPostprocessing(
                reason: "Pipeline produced a non-JSON-serializable result; ensure a stage populates context.json with JSON-valid values"
            )
        }
        let data = try JSONSerialization.data(withJSONObject: parsed)
        let rawJSON = String(data: data, encoding: .utf8)!

        let totalMs = context.trace.reduce(0) { $0 + $1.milliseconds }
        let metadata = ExtractionMetadata(
            modelName: context.userInfo["modelName"] as? String ?? "pipeline",
            inferenceTimeMs: totalMs,
            stages: context.trace
        )

        return ExtractionResult(
            json: parsed,
            rawJSON: rawJSON,
            rawOutput: context.intermediate ?? "",
            metadata: metadata
        )
    }

    /// Run the pipeline and decode the result into a `Decodable` type.
    public func run<T: Decodable>(on text: String, as type: T.Type) async throws -> TypedExtractionResult<T> {
        let result = try await run(on: text)
        let decoded = try JSONDecoder().decode(T.self, from: Data(result.rawJSON.utf8))
        return TypedExtractionResult(
            value: decoded,
            rawJSON: result.rawJSON,
            rawOutput: result.rawOutput,
            metadata: result.metadata
        )
    }
}

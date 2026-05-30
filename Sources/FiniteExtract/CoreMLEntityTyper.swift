import Foundation
import CoreML

/// Open Core ML NER runner for the typing pre-pass. Loads a GLiNER-family model
/// converted to Core ML and returns typed spans for caller-requested zero-shot labels.
///
/// The engine owns the model lifecycle and the generic, model-independent span
/// clean-up (drop lowercase-initial descriptor spans, drop pure breed/descriptor
/// words, strip a trailing "the {breed}" so "Bruno the Rottweiler" → "Bruno"). The
/// model-specific tokenize → infer → decode step is supplied as `decode`, because the
/// exact input/output feature contract is fixed only when the model is converted
/// (research-paper §19.4: GLiNER ONNX → Core ML). Until then, pass a `decode` closure
/// targeting your converted model; the rest of the pre-pass is reusable as-is.
///
/// `@unchecked Sendable`: the only non-Sendable member is `MLModel`, whose
/// `prediction(from:)` is documented thread-safe, and the typer is otherwise immutable.
public final class CoreMLEntityTyper: EntityTyper, @unchecked Sendable {
    /// Tokenize `text`, run `model` for `labels`, and return raw spans scoring ≥ `threshold`.
    public typealias Decode = @Sendable (_ text: String, _ model: MLModel, _ labels: [String], _ threshold: Double) throws -> [TypedSpan]

    public enum TyperError: Error, LocalizedError {
        /// No `decode` was supplied, so the model cannot be run yet (see §19.4).
        case decodeNotConfigured
        public var errorDescription: String? {
            switch self {
            case .decodeNotConfigured:
                "CoreMLEntityTyper has no decode closure; supply one for your converted GLiNER model (see §19.4)."
            }
        }
    }

    private let model: MLModel
    private let labels: [String]
    private let threshold: Double
    private let breeds: Set<String>
    private let decode: Decode?

    /// - Parameters:
    ///   - modelURL: a compiled Core ML model (`.mlmodelc`).
    ///   - labels: zero-shot labels to request, e.g. `["person", "pet"]`.
    ///   - threshold: minimum span score to keep.
    ///   - descriptorWords: lowercased words to reject as non-names (breeds, species).
    ///   - decode: model-specific tokenize + infer + decode. Required to run; when `nil`,
    ///     `typedSpans(in:)` throws `TyperError.decodeNotConfigured`.
    public init(
        modelURL: URL,
        labels: [String],
        threshold: Double = 0.40,
        descriptorWords: Set<String> = [],
        decode: Decode? = nil
    ) throws {
        self.model = try MLModel(contentsOf: modelURL)
        self.labels = labels
        self.threshold = threshold
        self.breeds = Set(descriptorWords.map { $0.lowercased() })
        self.decode = decode
    }

    public func typedSpans(in text: String) async throws -> [TypedSpan] {
        guard let decode else { throw TyperError.decodeNotConfigured }
        return try decode(text, model, labels, threshold).compactMap(postFilter)
    }

    /// Generic span clean-up shared with the Python `ner_prepass` (model-independent).
    private func postFilter(_ span: TypedSpan) -> TypedSpan? {
        let raw = span.text.trimmingCharacters(in: .whitespaces)
        guard let first = raw.first, first.isUppercase else { return nil }

        // "Bruno the Rottweiler" / "Penny a beagle" → keep the leading proper name.
        let name: String
        if let r = raw.range(of: "^[A-Z][a-zA-Z]+(?=\\s+(the|a|an)\\s)", options: .regularExpression) {
            name = String(raw[r])
        } else {
            name = raw
        }

        // Pure breed/descriptor word ("Husky", "Rottweiler") is not an entity name.
        if breeds.contains(name.lowercased()) { return nil }

        return TypedSpan(text: name, label: span.label, score: span.score, start: span.start, end: span.end)
    }
}

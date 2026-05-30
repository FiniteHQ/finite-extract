import Foundation

/// The result of an entity extraction operation.
///
/// `@unchecked Sendable` is safe here because the `json` dictionary is produced
/// by `JSONSerialization`, which returns only immutable Foundation value types.
/// The re-serialization step in `processRawOutput` guarantees this invariant.
public struct ExtractionResult: @unchecked Sendable {
    /// Parsed JSON output from the model (after repair and postprocessing).
    public let json: [String: Any]

    /// Cleaned JSON string (after repair and postprocessing).
    public let rawJSON: String

    /// Raw model output before any JSON extraction or repair.
    public let rawOutput: String

    /// Extraction metadata (latency, model info).
    public let metadata: ExtractionMetadata
}

/// A typed extraction result that decodes JSON into a Decodable type.
///
/// Returned by `extract(from:schema:as:)` for consumers who prefer
/// working with Codable types instead of `[String: Any]`.
public struct TypedExtractionResult<T: Decodable>: Sendable where T: Sendable {
    /// The decoded value.
    public let value: T

    /// Cleaned JSON string (after repair and postprocessing).
    public let rawJSON: String

    /// Raw model output before any JSON extraction or repair.
    public let rawOutput: String

    /// Extraction metadata (latency, model info).
    public let metadata: ExtractionMetadata
}

/// Metadata about an extraction operation.
public struct ExtractionMetadata: Sendable {
    /// Model used for extraction.
    public let modelName: String

    /// Total time in milliseconds. For a pipeline, the sum of all stage times.
    public let inferenceTimeMs: Int

    /// Per-stage timing breakdown when produced by an `ExtractionPipeline`.
    /// Empty for the legacy single-call helper path.
    public let stages: [StageTrace]

    public init(modelName: String, inferenceTimeMs: Int, stages: [StageTrace] = []) {
        self.modelName = modelName
        self.inferenceTimeMs = inferenceTimeMs
        self.stages = stages
    }
}

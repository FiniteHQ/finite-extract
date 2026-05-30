import Foundation
import MLXLLM
import MLXLMCommon

/// On-device structured entity extraction powered by MLX.
///
/// ```swift
/// let extractor = try await FiniteExtract()
///
/// let schema = ExtractionSchema(
///     name: "contacts",
///     systemPrompt: "Extract contact info from text. Return ONLY valid JSON...",
/// )
/// let result = try await extractor.extract(from: text, schema: schema)
/// print(result.json)
/// ```
public final class FiniteExtract: Sendable {
    private let container: ModelContainer

    /// The model identifier used for extraction (display name or HuggingFace ID).
    public let modelName: String

    /// Initialize with a known model configuration.
    ///
    /// Downloads model weights from HuggingFace on first use (~1-2 GB).
    /// Subsequent calls use the cached weights.
    ///
    /// - Parameters:
    ///   - model: The model to use for extraction. Defaults to `.recommended`.
    ///   - progressHandler: Optional callback for download progress.
    public init(
        model: ExtractModel = .recommended,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws {
        self.modelName = model.displayName

        let config = model.modelConfiguration
        let handler = progressHandler ?? { _ in }
        self.container = try await LLMModelFactory.shared.loadContainer(
            configuration: config,
            progressHandler: handler
        )
    }

    /// Initialize with a custom HuggingFace model ID.
    ///
    /// - Parameters:
    ///   - modelID: HuggingFace model repository ID (e.g. "mlx-community/Qwen2.5-3B-Instruct-4bit").
    ///   - progressHandler: Optional callback for download progress.
    public init(
        modelID: String,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws {
        self.modelName = modelID

        let config = ModelConfiguration(id: modelID)
        let handler = progressHandler ?? { _ in }
        self.container = try await LLMModelFactory.shared.loadContainer(
            configuration: config,
            progressHandler: handler
        )
    }

    /// Initialize from a local model directory — for a private model that isn't published
    /// to a HuggingFace repo. Loads weights directly from disk; no download, no token.
    ///
    /// - Parameters:
    ///   - modelDirectory: a directory containing the MLX model (`model.safetensors`,
    ///     `config.json`, tokenizer files).
    ///   - progressHandler: Optional callback (load is local, so typically a no-op).
    public init(
        modelDirectory: URL,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws {
        self.modelName = modelDirectory.lastPathComponent

        let config = ModelConfiguration(directory: modelDirectory)
        let handler = progressHandler ?? { _ in }
        self.container = try await LLMModelFactory.shared.loadContainer(
            configuration: config,
            progressHandler: handler
        )
    }

    /// Extract structured entities from text using the given schema.
    ///
    /// - Parameters:
    ///   - text: Free-form text to extract entities from.
    ///   - schema: The extraction schema defining prompts and output structure.
    ///   - maxTokens: Maximum tokens to generate. Defaults to 2048.
    ///   - temperature: Sampling temperature. Lower values (0.0-0.1) recommended for extraction. Defaults to 0.0.
    /// - Returns: The extraction result with parsed JSON, raw output, and metadata.
    /// - Note: The default models have a ~32K token context window. If the combined system prompt,
    ///   user prompt, and input text exceed this limit, the model may silently truncate input
    ///   and return partial extractions. For long documents, consider splitting the input.
    public func extract(
        from text: String,
        schema: ExtractionSchema,
        maxTokens: Int = 2048,
        temperature: Float = 0.0
    ) async throws -> ExtractionResult {
        // The single-pass path is the two-stage pipeline: generate, then JSON-repair.
        // Richer pipelines (typing pre-pass + NL intermediate + deterministic assembler)
        // are built directly with `ExtractionPipeline`.
        let pipeline = ExtractionPipeline([
            GenerateStage(
                self, systemPrompt: schema.systemPrompt,
                maxTokens: maxTokens, temperature: temperature,
                userPrompt: { [schema] text in schema.buildUserPrompt(from: text) }
            ),
            JSONRepairStage(postprocess: { [schema] (dict: inout [String: Any], input: String) in
                schema.applyPostprocessing(&dict, inputText: input)
            }),
        ])
        return try await pipeline.run(on: text)
    }

    /// Generate raw model text for a system + user prompt. The primitive `GenerateStage`
    /// builds on; exposed so custom stages and pipelines can drive the model directly.
    ///
    /// - Returns: The raw generated text and the inference time in milliseconds.
    public func generateText(
        systemPrompt: String,
        userPrompt: String,
        maxTokens: Int = 2048,
        temperature: Float = 0.0
    ) async throws -> (text: String, inferenceMs: Int) {
        try Task.checkCancellation()

        var params = GenerateParameters()
        params.maxTokens = maxTokens
        params.temperature = temperature

        let start = CFAbsoluteTimeGetCurrent()
        let session = ChatSession(container, instructions: systemPrompt, generateParameters: params)
        let rawOutput = try await session.respond(to: userPrompt)
        try Task.checkCancellation()

        return (rawOutput, Int((CFAbsoluteTimeGetCurrent() - start) * 1000))
    }

    /// Extract structured entities and decode into a Codable type.
    ///
    /// Convenience wrapper that extracts JSON and decodes it into `T` in one step,
    /// avoiding the need to manually re-serialize `[String: Any]` into your types.
    ///
    /// ```swift
    /// struct Contact: Decodable { let name: String; let email: String? }
    /// struct Contacts: Decodable { let contacts: [Contact] }
    ///
    /// let result = try await extractor.extract(from: text, schema: schema, as: Contacts.self)
    /// print(result.value.contacts)
    /// ```
    ///
    /// - Parameters:
    ///   - text: Free-form text to extract entities from.
    ///   - schema: The extraction schema defining prompts and output structure.
    ///   - type: The Decodable type to decode the JSON into.
    ///   - maxTokens: Maximum tokens to generate. Defaults to 2048.
    ///   - temperature: Sampling temperature. Defaults to 0.0.
    /// - Returns: A typed extraction result containing the decoded value and metadata.
    public func extract<T: Decodable>(
        from text: String,
        schema: ExtractionSchema,
        as type: T.Type,
        maxTokens: Int = 2048,
        temperature: Float = 0.0
    ) async throws -> TypedExtractionResult<T> {
        let result = try await extract(
            from: text,
            schema: schema,
            maxTokens: maxTokens,
            temperature: temperature
        )

        let data = Data(result.rawJSON.utf8)
        let decoded = try JSONDecoder().decode(T.self, from: data)

        return TypedExtractionResult(
            value: decoded,
            rawJSON: result.rawJSON,
            rawOutput: result.rawOutput,
            metadata: result.metadata
        )
    }

    // MARK: - Internal pipeline (testable without MLX)

    /// Process raw model output into an ExtractionResult.
    ///
    /// Handles JSON repair, postprocessing, and re-serialization.
    /// Exposed as internal static for unit testing without requiring a model.
    static func processRawOutput(
        _ rawOutput: String,
        schema: ExtractionSchema,
        inputText: String,
        modelName: String,
        inferenceMs: Int
    ) throws -> ExtractionResult {
        let (cleanJSON, parsed) = try repairAndPostprocess(
            rawOutput,
            postprocess: { (dict: inout [String: Any], input: String) in
                schema.applyPostprocessing(&dict, inputText: input)
            },
            inputText: inputText
        )
        let metadata = ExtractionMetadata(modelName: modelName, inferenceTimeMs: inferenceMs)
        return ExtractionResult(
            json: parsed,
            rawJSON: cleanJSON,
            rawOutput: rawOutput,
            metadata: metadata
        )
    }

    /// Repair model output into valid JSON, apply an optional postprocess closure, and
    /// re-serialize. Shared by `processRawOutput` and `JSONRepairStage` so both paths
    /// guarantee the same validity invariant.
    static func repairAndPostprocess(
        _ rawOutput: String,
        postprocess: (@Sendable (inout [String: Any], String) -> Void)?,
        inputText: String
    ) throws -> (cleanJSON: String, parsed: [String: Any]) {
        guard var (cleanJSON, parsed) = Postprocessor.extractJSON(from: rawOutput) else {
            throw ExtractionError.invalidJSON(rawOutput: rawOutput)
        }

        postprocess?(&parsed, inputText)

        guard JSONSerialization.isValidJSONObject(parsed) else {
            throw ExtractionError.invalidPostprocessing(
                reason: "Postprocessor inserted non-JSON-serializable values into the result dictionary"
            )
        }

        let reserialized = try JSONSerialization.data(withJSONObject: parsed)
        // JSONSerialization always produces valid UTF-8
        cleanJSON = String(data: reserialized, encoding: .utf8)!
        return (cleanJSON, parsed)
    }
}

/// Errors that can occur during extraction.
public enum ExtractionError: Error, LocalizedError {
    /// The model returned output that could not be parsed as valid JSON.
    case invalidJSON(rawOutput: String)

    /// The schema's postprocess closure inserted non-JSON-serializable values.
    case invalidPostprocessing(reason: String)

    public var errorDescription: String? {
        switch self {
        case .invalidJSON(let raw):
            "Model output is not valid JSON: \(raw.prefix(200))"
        case .invalidPostprocessing(let reason):
            "Postprocessing error: \(reason)"
        }
    }
}

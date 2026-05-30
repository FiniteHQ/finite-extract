import Foundation

/// Defines an extraction schema — the system prompt, user prompt construction,
/// and optional postprocessing that guide the model to produce structured JSON.
///
/// ```swift
/// let schema = ExtractionSchema(
///     name: "recipes",
///     systemPrompt: "Extract recipes from text. Return ONLY valid JSON matching: ...",
///     userPromptBuilder: { text in "Extract recipes from:\n\n\(text)" }
/// )
/// let result = try await extractor.extract(from: text, schema: schema)
/// ```
public struct ExtractionSchema: @unchecked Sendable {
    /// A short identifier for this schema (e.g., "recipes", "contacts").
    public let name: String

    /// The system prompt sent to the model. Should describe the extraction task,
    /// output format, and any rules for the model to follow.
    public let systemPrompt: String

    private let _userPromptBuilder: @Sendable (String) -> String
    private let _postprocess: (@Sendable (inout [String: Any], String) -> Void)?

    /// Create an extraction schema.
    ///
    /// - Parameters:
    ///   - name: Short identifier for this schema.
    ///   - systemPrompt: System prompt describing the extraction task and output format.
    ///   - userPromptBuilder: Closure that builds the user prompt from input text.
    ///     Defaults to wrapping the text with "Extract from the following text:".
    ///   - postprocess: Optional closure to fix up parsed JSON after extraction.
    ///     Receives the parsed dictionary (mutable) and the original input text.
    public init(
        name: String,
        systemPrompt: String,
        userPromptBuilder: @Sendable @escaping (String) -> String = { text in
            "Extract from the following text:\n\n\(text)"
        },
        postprocess: (@Sendable (inout [String: Any], String) -> Void)? = nil
    ) {
        self.name = name
        self.systemPrompt = systemPrompt
        self._userPromptBuilder = userPromptBuilder
        self._postprocess = postprocess
    }

    /// Build the user prompt from input text.
    public func buildUserPrompt(from text: String) -> String {
        _userPromptBuilder(text)
    }

    /// Apply schema-specific postprocessing to parsed JSON.
    public func applyPostprocessing(_ data: inout [String: Any], inputText: String) {
        _postprocess?(&data, inputText)
    }
}

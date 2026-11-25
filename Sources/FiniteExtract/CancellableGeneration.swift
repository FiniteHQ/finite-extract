import CoreGraphics
import Foundation
import MLX
import MLXLMCommon

// MARK: - The run loop

/// One generation run, split into the units of GPU work it submits.
///
/// `CancellableGeneration.run` owns every cancellation decision. A conforming type only does
/// the work it is asked to do, so the loop can be tested without a model.
protocol GenerationSteps {
    /// Prompt tokens that have not been prefilled yet.
    var pendingPromptTokens: Int { get }

    /// Run the model over the next `count` prompt tokens, writing them into the KV cache, and
    /// wait for that work to finish.
    mutating func prefill(count: Int)

    /// Hand the remaining prompt tokens (at most one chunk) to the token iterator. This submits
    /// one more chunk of work: those tokens plus the first sampled token.
    mutating func startDecoding(chunkSize: Int) throws

    /// Return the next token, or `nil` at the token limit. Submits the step for the token after it.
    mutating func nextToken() -> Int?

    /// Block until every piece of GPU work submitted so far has finished.
    func waitForSubmittedWork()
}

/// A point in a generation run, reported to a `GenerationProbe` (a test seam).
enum GenerationEvent: Equatable, Sendable {
    /// About to submit prefill chunk `index`, covering `tokens` prompt tokens.
    case prefillChunk(index: Int, tokens: Int)
    /// About to hand the last `tokens` prompt tokens to the token iterator.
    case decodeStart(tokens: Int)
    /// About to ask for token `index`, which submits the decode step after it (unless the
    /// token limit has been reached, when nothing is submitted).
    case decodeStep(index: Int)
    /// Token `index` has come back. Nothing new is submitted between this and the next step.
    case token(index: Int)

    /// Whether the run submits GPU work right after reporting this event.
    var submitsWork: Bool {
        switch self {
        case .prefillChunk, .decodeStart, .decodeStep: true
        case .token: false
        }
    }
}

/// Observes a generation run. Tests use it to count the work submitted after a cancel.
struct GenerationProbe: Sendable {
    let record: @Sendable (GenerationEvent) -> Void
}

enum CancellableGeneration {

    /// Drive `steps` from prompt to last token, checking for cancellation before every unit of
    /// GPU work.
    ///
    /// Once the calling task is cancelled, at most the unit already being submitted runs. The
    /// run then waits for the GPU to finish everything it was given and throws
    /// `CancellationError`, so nothing is still queued when this returns.
    ///
    /// The prompt is split the same way as mlx-swift-lm's `LLMModel.prepare`: whole chunks while
    /// more than one chunk is left, then the remainder (1 to `chunkSize` tokens) goes to the
    /// token iterator.
    ///
    /// - Parameter onToken: called with each token; return `false` to stop (end of sequence).
    static func run<Steps: GenerationSteps>(
        _ steps: inout Steps,
        prefillChunkSize chunkSize: Int,
        probe: GenerationProbe? = nil,
        onToken: (Int) -> Bool
    ) throws {
        do {
            var chunkIndex = 0
            while steps.pendingPromptTokens > chunkSize {
                try Task.checkCancellation()
                probe?.record(.prefillChunk(index: chunkIndex, tokens: chunkSize))
                steps.prefill(count: chunkSize)
                chunkIndex += 1
            }

            try Task.checkCancellation()
            probe?.record(.decodeStart(tokens: steps.pendingPromptTokens))
            try steps.startDecoding(chunkSize: chunkSize)

            var tokenIndex = 0
            while true {
                try Task.checkCancellation()
                probe?.record(.decodeStep(index: tokenIndex))
                guard let token = steps.nextToken() else { break }
                probe?.record(.token(index: tokenIndex))
                tokenIndex += 1
                if !onToken(token) { break }
            }
        } catch {
            steps.waitForSubmittedWork()
            throw error
        }
        // The token iterator always has the next step in flight; don't leave it running.
        steps.waitForSubmittedWork()
    }
}

// MARK: - The MLX steps

/// `GenerationSteps` over an mlx-swift-lm model, KV cache and `TokenIterator`.
///
/// Each method does exactly what mlx-swift-lm 2.31's own prefill and token loop do, so an
/// uncancelled run gives the same tokens as `ChatSession.respond` for the same chunk split.
struct MLXGenerationSteps: GenerationSteps {
    private let model: any LanguageModel
    private let cache: [KVCache]
    private let parameters: GenerateParameters
    private var pending: LMInput.Text
    private var iterator: TokenIterator?

    init(model: any LanguageModel, input: LMInput, parameters: GenerateParameters) {
        self.model = model
        self.cache = model.newCache(parameters: parameters)
        self.parameters = parameters
        self.pending = input.text
    }

    var pendingPromptTokens: Int { pending.tokens.size }

    mutating func prefill(count: Int) {
        // One iteration of `LLMModel.prepare(_:cache:windowSize:)`.
        let chunk = pending[.newAxis, ..<count]
        _ = model(chunk, cache: cache.isEmpty ? nil : cache, state: nil)
        eval(cache)
        pending = pending[count...]
    }

    mutating func startDecoding(chunkSize: Int) throws {
        // At most `chunkSize` tokens are left, so the iterator's own prefill loop has nothing
        // to do: it runs these tokens and samples the first token in one step.
        var parameters = parameters
        parameters.prefillStepSize = chunkSize
        iterator = try TokenIterator(
            input: LMInput(text: pending), model: model, cache: cache, parameters: parameters
        )
    }

    mutating func nextToken() -> Int? {
        iterator?.next()
    }

    func waitForSubmittedWork() {
        Stream().synchronize()
    }
}

// MARK: - Text

/// Turns tokens into the response text exactly as `ChatSession.respond` does in mlx-swift-lm
/// 2.31: stop at an end-of-sequence or unknown token, detokenize as a stream, and drop any text
/// the tool-call parser claims (ChatSession drops tool calls when no tool handler is set).
struct ResponseText {
    private var detokenizer: NaiveStreamingDetokenizer
    private let toolCallProcessor: ToolCallProcessor
    private let stopTokens: Set<Int>
    private(set) var text = ""

    init(context: ModelContext) {
        let tokenizer = context.tokenizer
        let configuration = context.configuration
        detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        toolCallProcessor = ToolCallProcessor(format: configuration.toolCallFormat ?? .json)

        var stopTokens = configuration.eosTokenIds
        if let eos = tokenizer.eosTokenId {
            stopTokens.insert(eos)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokens.insert(id)
            }
        }
        if let unknown = tokenizer.unknownTokenId {
            stopTokens.insert(unknown)
        }
        self.stopTokens = stopTokens
    }

    /// Add one token. Returns `false` when the token ends the response.
    mutating func append(_ token: Int) -> Bool {
        if stopTokens.contains(token) {
            return false
        }
        detokenizer.append(token: token)
        if let chunk = detokenizer.next() {
            if let visible = toolCallProcessor.processChunk(chunk) {
                text += visible
            }
            _ = toolCallProcessor.toolCalls.popLast()
        }
        return true
    }
}

// MARK: - Entry point

extension CancellableGeneration {

    /// Generate the response to `chat` with the model in `container`.
    static func respond(
        container: ModelContainer,
        chat: [Chat.Message],
        parameters: GenerateParameters,
        prefillChunkSize: Int,
        probe: GenerationProbe?
    ) async throws -> String {
        // The same input ChatSession builds (its default media processing included).
        let userInput = UserInput(
            chat: chat,
            processing: .init(resize: CGSize(width: 512, height: 512))
        )
        let input = try await container.prepare(input: userInput)

        // Like ChatSession, run outside the container's lock so separate calls can overlap as
        // they did in 0.1.1. The KV cache belongs to this call alone.
        let loaded = await container.perform { context in
            Unchecked((context.model, ResponseText(context: context)))
        }
        let (model, emptyResponse) = loaded.value
        var response = emptyResponse
        var steps = MLXGenerationSteps(model: model, input: input, parameters: parameters)
        try run(&steps, prefillChunkSize: prefillChunkSize, probe: probe) { token in
            response.append(token)
        }
        return response.text
    }
}

/// Carries model-owned values out of `ModelContainer.perform`, as ChatSession does. The values
/// are only used by the one task that made them.
private struct Unchecked<Value>: @unchecked Sendable {
    let value: Value
    init(_ value: Value) { self.value = value }
}

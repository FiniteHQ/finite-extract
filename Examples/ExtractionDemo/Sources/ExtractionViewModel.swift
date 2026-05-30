import Foundation
import FiniteExtract

@MainActor
@Observable
final class ExtractionViewModel {
    var systemPrompt: String = """
    Extract contacts from text. Return ONLY valid JSON.
    Format: {"contacts": [{"name": "...", "email": "...", "role": "...", "company": "..."}]}
    Only include fields present in the text. Use null for missing fields.
    """

    var inputText: String = ""
    var resultJSON: String = ""
    var inferenceMs: Int = 0
    var rawOutput: String = ""

    var selectedModel: ExtractModel = .qwen2_5_3b
    var phase: Phase = .needsDownload

    var downloadProgress: Double = 0

    enum Phase: Equatable {
        case needsDownload
        case downloading
        case ready
        case extracting
        case done
        case error(String)
    }

    private var extractor: FiniteExtract?

    var canExtract: Bool {
        extractor != nil && phase != .extracting && phase != .downloading && phase != .needsDownload
    }

    func loadModel() async {
        phase = .downloading
        downloadProgress = 0

        do {
            extractor = try await FiniteExtract(model: selectedModel) { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress.fractionCompleted
                }
            }
            phase = .ready
        } catch {
            phase = .error("Download failed: \(error.localizedDescription)")
        }
    }

    func switchModel(_ model: ExtractModel) {
        guard model != selectedModel else { return }
        selectedModel = model
        extractor = nil
        phase = .needsDownload
        resultJSON = ""
        rawOutput = ""
    }

    func extract() async {
        guard let extractor, !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }

        phase = .extracting
        resultJSON = ""
        rawOutput = ""

        let schema = ExtractionSchema(
            name: "demo",
            systemPrompt: systemPrompt
        )

        do {
            let result = try await extractor.extract(
                from: inputText,
                schema: schema
            )

            inferenceMs = result.metadata.inferenceTimeMs
            rawOutput = result.rawOutput

            // Pretty-print
            if let data = try? JSONSerialization.data(withJSONObject: result.json, options: [.prettyPrinted, .sortedKeys]),
               let pretty = String(data: data, encoding: .utf8) {
                resultJSON = pretty
            } else {
                resultJSON = result.rawJSON
            }

            phase = .done
        } catch {
            rawOutput = (error as? ExtractionError).map {
                if case .invalidJSON(let raw) = $0 { return raw }
                return ""
            } ?? ""
            phase = .error(error.localizedDescription)
        }
    }
}

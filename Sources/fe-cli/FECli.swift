import Foundation
import FiniteExtract

@main
struct FECli {
    static func main() async throws {
        let args = CommandLine.arguments

        var systemPrompt: String? = nil
        var promptFilePath: String? = nil
        var inputText: String? = nil
        var modelName: String? = nil

        var i = 1
        while i < args.count {
            switch args[i] {
            case "--prompt", "-p":
                i += 1
                guard i < args.count else {
                    print("Error: --prompt requires a value")
                    Foundation.exit(1)
                }
                systemPrompt = args[i]
            case "--prompt-file":
                i += 1
                guard i < args.count else {
                    print("Error: --prompt-file requires a value")
                    Foundation.exit(1)
                }
                promptFilePath = args[i]
            case "--text", "-t":
                i += 1
                guard i < args.count else {
                    print("Error: --text requires a value")
                    Foundation.exit(1)
                }
                inputText = args[i]
            case "--model", "-m":
                i += 1
                guard i < args.count else {
                    print("Error: --model requires a value")
                    Foundation.exit(1)
                }
                modelName = args[i]
            case "--help", "-h":
                printUsage()
                return
            default:
                break
            }
            i += 1
        }

        // Load system prompt from file if specified
        if let path = promptFilePath {
            if systemPrompt != nil {
                print("Error: --prompt and --prompt-file are mutually exclusive")
                Foundation.exit(1)
            }
            let url = URL(fileURLWithPath: path)
            systemPrompt = try String(contentsOf: url, encoding: .utf8)
        }

        guard let prompt = systemPrompt else {
            print("Error: --prompt or --prompt-file is required")
            printUsage()
            Foundation.exit(1)
        }

        // Read text from stdin if not provided via --text
        let text: String
        if let t = inputText {
            text = t
        } else {
            print("Reading from stdin (Ctrl+D to finish)...")
            var lines: [String] = []
            while let line = readLine(strippingNewline: false) {
                lines.append(line)
            }
            text = lines.joined()
            guard !text.isEmpty else {
                print("Error: no input text provided")
                Foundation.exit(1)
            }
        }

        // Load model
        let model: ExtractModel = (modelName == "1.5b") ? .qwen2_5_1_5b : .qwen2_5_3b
        print("Loading model (\(model.displayName))...")
        let extractor = try await FiniteExtract(model: model) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            if pct % 10 == 0 {
                print("  Download: \(pct)%", terminator: "\r")
                fflush(stdout)
            }
        }

        let schema = ExtractionSchema(name: "cli", systemPrompt: prompt)

        print("Extracting...")
        let result = try await extractor.extract(from: text, schema: schema)

        // Pretty-print the JSON
        if let data = try? JSONSerialization.data(withJSONObject: result.json, options: [.prettyPrinted, .sortedKeys]),
           let pretty = String(data: data, encoding: .utf8) {
            print("\n\(pretty)")
        } else {
            print("\n\(result.rawJSON)")
        }

        print("\nInference: \(result.metadata.inferenceTimeMs)ms")
    }

    static func printUsage() {
        print("""
        fe-cli - FiniteExtract CLI

        Usage: fe-cli --prompt "..." --text "..." [options]
               echo "text" | fe-cli --prompt-file prompt.txt

        Options:
          -p, --prompt <text>       System prompt for extraction
              --prompt-file <path>  Load system prompt from file
          -t, --text <text>         Input text (reads stdin if omitted)
          -m, --model <name>        Model: "3b" (default) or "1.5b"
          -h, --help                Show this help
        """)
    }
}

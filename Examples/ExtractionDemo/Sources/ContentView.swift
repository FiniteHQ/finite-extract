import SwiftUI
import FiniteExtract

struct ContentView: View {
    @State private var vm = ExtractionViewModel()
    @State private var showRawOutput = false

    var body: some View {
        NavigationStack {
            Group {
                switch vm.phase {
                case .needsDownload:
                    downloadPromptView
                case .downloading:
                    downloadProgressView
                default:
                    extractionView
                }
            }
            .navigationTitle("finite-extract demo")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    modelPicker
                }
            }
        }
    }

    // MARK: - Download

    private var downloadPromptView: some View {
        VStack(spacing: 20) {
            Spacer()

            Image(systemName: "arrow.down.circle")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)

            Text("Download \(vm.selectedModel.displayName)")
                .font(.title2.bold())

            Text("~\(vm.selectedModel.approximateSizeMB / 1000) GB one-time download.\nThe model runs entirely on-device.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            Button("Download Model") {
                Task { await vm.loadModel() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
        }
        .padding()
    }

    private var downloadProgressView: some View {
        VStack(spacing: 16) {
            Spacer()

            ProgressView(value: vm.downloadProgress) {
                Text("Downloading model...")
                    .font(.headline)
            } currentValueLabel: {
                Text("\(Int(vm.downloadProgress * 100))%")
                    .monospacedDigit()
            }
            .padding(.horizontal, 40)

            Text("This is a one-time download.\nThe model runs entirely on-device.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Spacer()
        }
        .padding()
    }

    // MARK: - Extraction

    private var extractionView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                sectionHeader("System Prompt")
                TextEditor(text: $vm.systemPrompt)
                    .font(.system(.caption, design: .monospaced))
                    .frame(minHeight: 100, maxHeight: 140)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(.quaternary)
                    )

                sectionHeader("Input Text")
                TextEditor(text: $vm.inputText)
                    .font(.body)
                    .frame(minHeight: 80, maxHeight: 120)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(.quaternary)
                    )
                    .overlay(alignment: .topLeading, content: {
                        if vm.inputText.isEmpty {
                            Text("Paste or type text to extract from...")
                                .foregroundStyle(.tertiary)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 8)
                                .allowsHitTesting(false)
                        }
                    })

                extractButton

                if !vm.resultJSON.isEmpty || vm.phase == .extracting || isError {
                    resultView
                }
            }
            .padding()
        }
        .scrollDismissesKeyboard(.interactively)
    }

    private var extractButton: some View {
        Button {
            Task { await vm.extract() }
        } label: {
            HStack {
                if vm.phase == .extracting {
                    ProgressView()
                        .controlSize(.small)
                    Text("Extracting...")
                } else {
                    Text("Extract")
                }
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(!vm.canExtract || vm.inputText.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    @ViewBuilder
    private var resultView: some View {
        HStack {
            sectionHeader("Result")
            Spacer()
            if vm.inferenceMs > 0 && !vm.resultJSON.isEmpty {
                Text(formatTime(vm.inferenceMs))
                    .font(.caption)
                    .monospacedDigit()
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.fill.tertiary, in: Capsule())
            }
        }

        if vm.phase == .extracting {
            HStack {
                ProgressView()
                    .controlSize(.small)
                Text("Thinking...")
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 8)
        } else if case .error(let msg) = vm.phase {
            VStack(alignment: .leading, spacing: 8) {
                Label(msg, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.caption)

                if !vm.rawOutput.isEmpty {
                    DisclosureGroup("Raw model output") {
                        Text(vm.rawOutput)
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.caption)
                }
            }
        } else if !vm.resultJSON.isEmpty {
            Text(vm.resultJSON)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(.fill.quinary, in: RoundedRectangle(cornerRadius: 8))

            if !vm.rawOutput.isEmpty {
                DisclosureGroup("Raw model output") {
                    Text(vm.rawOutput)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Components

    private var modelPicker: some View {
        Menu {
            ForEach(ExtractModel.allCases, id: \.self) { model in
                Button {
                    vm.switchModel(model)
                } label: {
                    HStack {
                        Text(model.displayName)
                        if model == vm.selectedModel {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            Text(shortModelName)
                .font(.caption.bold())
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.fill.tertiary, in: Capsule())
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.caption)
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }

    // MARK: - Helpers

    private var shortModelName: String {
        switch vm.selectedModel {
        case .qwen2_5_3b: "3B"
        case .qwen2_5_1_5b: "1.5B"
        }
    }

    private var isError: Bool {
        if case .error = vm.phase { return true }
        return false
    }

    private func formatTime(_ ms: Int) -> String {
        if ms < 1000 { return "\(ms)ms" }
        return String(format: "%.1fs", Double(ms) / 1000)
    }
}

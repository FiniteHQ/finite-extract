import Foundation
import MLXLMCommon

/// Known model configurations for entity extraction.
///
/// Each entry maps to a HuggingFace MLX model repo with settings tuned
/// for structured JSON extraction tasks. Based on benchmark results,
/// qwen2.5:3b is the recommended default (best extraction quality).
public enum ExtractModel: String, Sendable, CaseIterable {
    case qwen2_5_3b = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    case qwen2_5_1_5b = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

    /// The recommended model for best extraction quality.
    public static let recommended: ExtractModel = .qwen2_5_3b

    /// HuggingFace model ID.
    public var id: String { rawValue }

    /// Human-readable display name.
    public var displayName: String {
        switch self {
        case .qwen2_5_3b: "Qwen 2.5 3B Instruct"
        case .qwen2_5_1_5b: "Qwen 2.5 1.5B Instruct"
        }
    }

    /// Approximate download size in MB (4-bit quantized).
    public var approximateSizeMB: Int {
        switch self {
        case .qwen2_5_3b: 2000
        case .qwen2_5_1_5b: 1000
        }
    }

    /// Create a ModelConfiguration suitable for MLX Swift LM.
    public var modelConfiguration: ModelConfiguration {
        ModelConfiguration(id: id)
    }
}

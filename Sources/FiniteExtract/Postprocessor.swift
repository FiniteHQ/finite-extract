import Foundation

/// Repairs and validates JSON output from small language models.
///
/// Small models sometimes produce almost-valid JSON. This pipeline handles
/// common issues: markdown fences, trailing commas, escaped quotes, and
/// template contamination.
public enum Postprocessor {
    /// Attempt to extract valid JSON from raw model output.
    ///
    /// Handles: markdown code fences, leading/trailing text, trailing commas,
    /// escaped single quotes, and template contamination in source_text fields.
    ///
    /// - Parameter raw: Raw string output from the model.
    /// - Returns: The cleaned JSON string and parsed dictionary, or nil if unrecoverable.
    ///   If the model returns a JSON array, it is wrapped as `{"items": [...]}`.
    public static func extractJSON(from raw: String) -> (string: String, parsed: [String: Any])? {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)

        // Try as-is first
        if let result = tryParse(s) {
            return result
        }

        // Strip markdown code fences
        if let openRange = s.range(of: "```(?:json)?\\s*\n?", options: .regularExpression),
           let closeRange = s.range(of: "\n?```", options: .regularExpression, range: openRange.upperBound..<s.endIndex) {
            let inner = String(s[openRange.upperBound..<closeRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let result = tryParse(inner) {
                return result
            }
            s = inner
        }

        // Find first { and last } (or [ and ] for arrays)
        let dictStart = s.firstIndex(of: "{")
        let dictEnd = s.lastIndex(of: "}")
        let arrStart = s.firstIndex(of: "[")
        let arrEnd = s.lastIndex(of: "]")

        let start: String.Index
        let end: String.Index
        if let ds = dictStart, let de = dictEnd, ds < de {
            if let as_ = arrStart, as_ < ds, let ae = arrEnd, ae > de {
                start = as_; end = ae
            } else {
                start = ds; end = de
            }
        } else if let as_ = arrStart, let ae = arrEnd, as_ < ae {
            start = as_; end = ae
        } else {
            return nil
        }
        let candidate = String(s[start...end])
        if let result = tryParse(candidate) {
            return result
        }

        // Fix trailing commas before } or ]
        var fixed = candidate.replacingOccurrences(
            of: ",\\s*([}\\]])",
            with: "$1",
            options: .regularExpression
        )
        if let result = tryParse(fixed) {
            return result
        }

        // Fix escaped single quotes (\' is invalid in JSON, replace with ')
        fixed = fixed.replacingOccurrences(of: "\\'", with: "'")
        if let result = tryParse(fixed) {
            return result
        }

        // Fix template contamination: "source_text":<the note>" or "source_text": <the note>
        fixed = fixed.replacingOccurrences(
            of: "\"source_text\":\\s*<[^>]*>\"?",
            with: "\"source_text\": \"\"",
            options: .regularExpression
        )
        if let result = tryParse(fixed) {
            return result
        }

        return nil
    }

    private static func tryParse(_ s: String) -> (string: String, parsed: [String: Any])? {
        guard let data = s.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }
        if let dict = obj as? [String: Any] {
            return (s, dict)
        }
        if let arr = obj as? [Any] {
            // Wrap arrays so callers always get a dictionary
            let wrapped: [String: Any] = ["items": arr]
            if let wrappedData = try? JSONSerialization.data(withJSONObject: wrapped),
               let wrappedString = String(data: wrappedData, encoding: .utf8) {
                return (wrappedString, wrapped)
            }
        }
        return nil
    }
}

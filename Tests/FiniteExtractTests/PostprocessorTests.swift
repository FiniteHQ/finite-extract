import XCTest
@testable import FiniteExtract

final class PostprocessorTests: XCTestCase {

    // MARK: - JSON repair

    func testValidJSON() {
        let input = """
        {"name": "Sarah", "age": 30}
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["name"] as? String, "Sarah")
    }

    func testMarkdownFences() {
        let input = """
        ```json
        {"name": "Tom", "pet": "Biscuit"}
        ```
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["name"] as? String, "Tom")
    }

    func testLeadingText() {
        let input = """
        Here is the extracted data:
        {"version": "v1", "confidence": 0.9}
        Some trailing text
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["version"] as? String, "v1")
    }

    func testTrailingComma() {
        let input = """
        {"name": "Sarah", "age": 30,}
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["name"] as? String, "Sarah")
    }

    func testCompletelyInvalid() {
        let result = Postprocessor.extractJSON(from: "not json at all")
        XCTAssertNil(result)
    }

    func testEscapedSingleQuote() {
        let input = """
        {"note": "she\\'s doing well"}
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["note"] as? String, "she's doing well")
    }

    func testTemplateContamination() {
        let input = """
        {"version": "v1", "source_text":<the note>", "confidence": 0.9}
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["version"] as? String, "v1")
    }

    func testEmptyInput() {
        XCTAssertNil(Postprocessor.extractJSON(from: ""))
    }

    func testNestedTrailingCommas() {
        let input = """
        {"people": [{"name": "Sarah",}, {"name": "Tom",},],}
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        let people = result?.parsed["people"] as? [[String: Any]]
        XCTAssertEqual(people?.count, 2)
    }

    func testMarkdownFencesWithoutJsonLabel() {
        let input = """
        ```
        {"name": "unlabeled"}
        ```
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["name"] as? String, "unlabeled")
    }

    // MARK: - Schema

    func testSchemaCreation() {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract data. Return JSON."
        )
        XCTAssertEqual(schema.name, "test")
        XCTAssertEqual(schema.systemPrompt, "Extract data. Return JSON.")
    }

    func testSchemaUserPromptBuilder() {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            userPromptBuilder: { "INPUT: \($0)" }
        )
        XCTAssertEqual(schema.buildUserPrompt(from: "hello"), "INPUT: hello")
    }

    func testSchemaDefaultUserPromptBuilder() {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let prompt = schema.buildUserPrompt(from: "hello")
        XCTAssertTrue(prompt.contains("hello"))
    }

    func testSchemaPostprocessing() {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, _ in
                data["processed"] = true
            }
        )
        var data: [String: Any] = ["value": 1]
        schema.applyPostprocessing(&data, inputText: "")
        XCTAssertEqual(data["processed"] as? Bool, true)
    }

    func testSchemaNoPostprocessing() {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        var data: [String: Any] = ["value": 1]
        schema.applyPostprocessing(&data, inputText: "")
        XCTAssertEqual(data.count, 1) // unchanged
    }

    func testMultilineMarkdownFences() {
        let input = """
        ```json
        {
            "name": "Sarah",
            "age": 30,
            "email": "sarah@example.com"
        }
        ```
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.parsed["name"] as? String, "Sarah")
        XCTAssertEqual(result?.parsed["age"] as? Int, 30)
    }

    func testJSONArray() {
        let input = """
        [{"name": "Sarah"}, {"name": "Tom"}]
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        let items = result?.parsed["items"] as? [[String: Any]]
        XCTAssertEqual(items?.count, 2)
        XCTAssertEqual(items?.first?["name"] as? String, "Sarah")
    }

    func testJSONArrayInFences() {
        let input = """
        ```json
        [{"name": "A"}, {"name": "B"}]
        ```
        """
        let result = Postprocessor.extractJSON(from: input)
        XCTAssertNotNil(result)
        let items = result?.parsed["items"] as? [[String: Any]]
        XCTAssertEqual(items?.count, 2)
    }

    // MARK: - Model registry

    func testModelRegistry() {
        XCTAssertEqual(ExtractModel.recommended, .qwen2_5_3b)
        XCTAssertFalse(ExtractModel.qwen2_5_3b.id.isEmpty)
        XCTAssertFalse(ExtractModel.qwen2_5_3b.displayName.isEmpty)
        XCTAssertGreaterThan(ExtractModel.qwen2_5_3b.approximateSizeMB, 0)
        XCTAssertEqual(ExtractModel.allCases.count, 2)
    }
}

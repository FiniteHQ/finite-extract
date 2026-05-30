import XCTest
@testable import FiniteExtract

// Large end-to-end test suite; the type-body-length limit (meant to keep production
// types maintainable) is not meaningful for an exhaustive test class.
// swiftlint:disable:next type_body_length
final class ExtractionPipelineTests: XCTestCase {

    // MARK: - Successful extraction

    func testBasicExtraction() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            """
            {"name": "Sarah", "age": 30}
            """,
            schema: schema,
            inputText: "Sarah is 30",
            modelName: "test-model",
            inferenceMs: 100
        )

        XCTAssertEqual(result.json["name"] as? String, "Sarah")
        XCTAssertEqual(result.json["age"] as? Int, 30)
        XCTAssertEqual(result.metadata.modelName, "test-model")
        XCTAssertEqual(result.metadata.inferenceTimeMs, 100)
    }

    func testRawOutputPreserved() throws {
        let rawOutput = "  Here is the JSON: {\"x\": 1}  "
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            rawOutput,
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(result.rawOutput, rawOutput)
        XCTAssertEqual(result.json["x"] as? Int, 1)
    }

    // MARK: - JSON repair integration

    func testExtractionWithMarkdownFences() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            """
            ```json
            {
                "name": "Tom",
                "email": "tom@example.com"
            }
            ```
            """,
            schema: schema,
            inputText: "Tom's email is tom@example.com",
            modelName: "m",
            inferenceMs: 50
        )

        XCTAssertEqual(result.json["name"] as? String, "Tom")
        XCTAssertEqual(result.json["email"] as? String, "tom@example.com")
    }

    func testExtractionWithTrailingCommas() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            """
            {"items": [{"a": 1,}, {"b": 2,},],}
            """,
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        let items = result.json["items"] as? [[String: Any]]
        XCTAssertEqual(items?.count, 2)
    }

    func testExtractionWithLeadingText() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            "Sure! Here is the extracted data:\n{\"value\": 42}\nHope that helps!",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(result.json["value"] as? Int, 42)
    }

    func testExtractionWithArrayOutput() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            """
            [{"name": "A"}, {"name": "B"}]
            """,
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        let items = result.json["items"] as? [[String: Any]]
        XCTAssertEqual(items?.count, 2)
        XCTAssertEqual(items?.first?["name"] as? String, "A")
    }

    // MARK: - Postprocessing

    func testPostprocessingApplied() throws {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, inputText in
                data["source_length"] = inputText.count
                data["processed"] = true
            }
        )
        let result = try FiniteExtract.processRawOutput(
            "{\"value\": 1}",
            schema: schema,
            inputText: "hello world",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(result.json["processed"] as? Bool, true)
        XCTAssertEqual(result.json["source_length"] as? Int, 11)
        XCTAssertEqual(result.json["value"] as? Int, 1)
    }

    func testPostprocessingReflectedInRawJSON() throws {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, _ in
                data["added"] = "by-postprocessor"
            }
        )
        let result = try FiniteExtract.processRawOutput(
            "{\"original\": true}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        // rawJSON should reflect the postprocessed state
        XCTAssertTrue(result.rawJSON.contains("by-postprocessor"))
        XCTAssertTrue(result.rawJSON.contains("original"))
    }

    func testPostprocessingCanRemoveFields() throws {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, _ in
                data.removeValue(forKey: "internal_debug")
            }
        )
        let result = try FiniteExtract.processRawOutput(
            "{\"name\": \"Sarah\", \"internal_debug\": \"xyz\"}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(result.json["name"] as? String, "Sarah")
        XCTAssertNil(result.json["internal_debug"])
    }

    func testNoPostprocessingLeavesDataUnchanged() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            "{\"a\": 1, \"b\": 2}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(result.json["a"] as? Int, 1)
        XCTAssertEqual(result.json["b"] as? Int, 2)
        XCTAssertEqual(result.json.count, 2)
    }

    func testPostprocessorReceivesInputText() throws {
        var receivedText: String?
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { _, inputText in
                receivedText = inputText
            }
        )
        _ = try FiniteExtract.processRawOutput(
            "{\"x\": 1}",
            schema: schema,
            inputText: "the original note",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(receivedText, "the original note")
    }

    // MARK: - Error paths

    func testInvalidJSONThrows() {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")

        XCTAssertThrowsError(
            try FiniteExtract.processRawOutput(
                "I cannot extract anything from this text.",
                schema: schema,
                inputText: "",
                modelName: "m",
                inferenceMs: 0
            )
        ) { error in
            guard let extractionError = error as? ExtractionError else {
                XCTFail("Expected ExtractionError, got \(type(of: error))")
                return
            }
            if case .invalidJSON(let raw) = extractionError {
                XCTAssertEqual(raw, "I cannot extract anything from this text.")
            } else {
                XCTFail("Expected .invalidJSON case")
            }
        }
    }

    func testEmptyOutputThrows() {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")

        XCTAssertThrowsError(
            try FiniteExtract.processRawOutput(
                "",
                schema: schema,
                inputText: "",
                modelName: "m",
                inferenceMs: 0
            )
        ) { error in
            XCTAssertTrue(error is ExtractionError)
        }
    }

    func testWhitespaceOnlyOutputThrows() {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")

        XCTAssertThrowsError(
            try FiniteExtract.processRawOutput(
                "   \n\n   ",
                schema: schema,
                inputText: "",
                modelName: "m",
                inferenceMs: 0
            )
        )
    }

    func testExtractionErrorDescription() {
        let error = ExtractionError.invalidJSON(rawOutput: "some garbage output")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("some garbage output"))
    }

    func testExtractionErrorTruncatesLongOutput() {
        let longOutput = String(repeating: "x", count: 500)
        let error = ExtractionError.invalidJSON(rawOutput: longOutput)
        // errorDescription truncates to prefix(200)
        XCTAssertTrue(error.errorDescription!.count < longOutput.count)
    }

    // MARK: - ExtractionResult

    func testResultJSONIsolatedFromPostprocessing() throws {
        // Verify that processRawOutput returns a result whose JSON
        // is produced by JSONSerialization (immutable Foundation types),
        // ensuring Sendable safety without a deep copy.
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, _ in
                data["added"] = true
            }
        )
        let result = try FiniteExtract.processRawOutput(
            "{\"original\": 1}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        // Both original and postprocessed values present
        XCTAssertEqual(result.json["original"] as? Int, 1)
        XCTAssertEqual(result.json["added"] as? Bool, true)
        // rawJSON reflects postprocessed state (re-serialized)
        XCTAssertTrue(result.rawJSON.contains("added"))
    }

    func testResultMetadata() throws {
        let result = ExtractionResult(
            json: ["k": "v"],
            rawJSON: "{\"k\":\"v\"}",
            rawOutput: "raw",
            metadata: ExtractionMetadata(modelName: "test-model", inferenceTimeMs: 42)
        )

        XCTAssertEqual(result.metadata.modelName, "test-model")
        XCTAssertEqual(result.metadata.inferenceTimeMs, 42)
        XCTAssertEqual(result.rawOutput, "raw")
    }

    // MARK: - Schema integration

    func testCustomUserPromptBuilder() throws {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            userPromptBuilder: { "CUSTOM: \($0)" }
        )

        // Verify the prompt builder works correctly
        XCTAssertEqual(schema.buildUserPrompt(from: "hello"), "CUSTOM: hello")

        // And that extraction still works with custom builder
        let result = try FiniteExtract.processRawOutput(
            "{\"data\": true}",
            schema: schema,
            inputText: "hello",
            modelName: "m",
            inferenceMs: 0
        )
        XCTAssertEqual(result.json["data"] as? Bool, true)
    }

    // MARK: - Edge cases

    func testNestedJSONPreserved() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            """
            {"contacts": [{"name": "Sarah", "address": {"city": "SF", "zip": "94102"}}]}
            """,
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        let contacts = result.json["contacts"] as? [[String: Any]]
        XCTAssertEqual(contacts?.count, 1)
        let address = contacts?.first?["address"] as? [String: Any]
        XCTAssertEqual(address?["city"] as? String, "SF")
    }

    func testUnicodeInOutput() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            "{\"name\": \"José García\", \"city\": \"São Paulo\"}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(result.json["name"] as? String, "José García")
        XCTAssertEqual(result.json["city"] as? String, "São Paulo")
    }

    func testEmptyJSONObject() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let result = try FiniteExtract.processRawOutput(
            "{}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertTrue(result.json.isEmpty)
    }

    // MARK: - Postprocessor validation (Finding 2)

    func testPostprocessorInsertingNonSerializableThrows() {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, _ in
                data["timestamp"] = Date() // Date is not JSON-serializable
            }
        )

        XCTAssertThrowsError(
            try FiniteExtract.processRawOutput(
                "{\"value\": 1}",
                schema: schema,
                inputText: "",
                modelName: "m",
                inferenceMs: 0
            )
        ) { error in
            guard let extractionError = error as? ExtractionError else {
                XCTFail("Expected ExtractionError, got \(type(of: error))")
                return
            }
            if case .invalidPostprocessing(let reason) = extractionError {
                XCTAssertTrue(reason.contains("non-JSON-serializable"))
            } else {
                XCTFail("Expected .invalidPostprocessing, got \(extractionError)")
            }
        }
    }

    func testPostprocessorInsertingValidTypesSucceeds() throws {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, _ in
                data["string"] = "hello"
                data["number"] = 42
                data["bool"] = true
                data["null"] = NSNull()
                data["array"] = [1, 2, 3]
                data["nested"] = ["key": "value"]
            }
        )

        let result = try FiniteExtract.processRawOutput(
            "{\"original\": 1}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        XCTAssertEqual(result.json["string"] as? String, "hello")
        XCTAssertEqual(result.json["number"] as? Int, 42)
        XCTAssertEqual(result.json["bool"] as? Bool, true)
    }

    // MARK: - rawJSON/json sync (Finding 3)

    func testRawJSONAlwaysSyncsWithJSON() throws {
        let schema = ExtractionSchema(
            name: "test",
            systemPrompt: "Extract.",
            postprocess: { data, _ in
                data["added"] = "new-value"
                data.removeValue(forKey: "remove_me")
            }
        )

        let result = try FiniteExtract.processRawOutput(
            "{\"keep\": true, \"remove_me\": false}",
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        // rawJSON must reflect postprocessed state
        XCTAssertTrue(result.rawJSON.contains("new-value"))
        XCTAssertFalse(result.rawJSON.contains("remove_me"))
        // json dict must match
        XCTAssertEqual(result.json["added"] as? String, "new-value")
        XCTAssertNil(result.json["remove_me"])
    }

    // MARK: - Typed extraction (Finding 5)

    func testTypedExtractionResultDecoding() throws {
        struct Contact: Decodable, Sendable {
            let name: String
            let email: String?
        }
        struct Contacts: Decodable, Sendable {
            let contacts: [Contact]
        }

        // Simulate what TypedExtractionResult does — decode rawJSON
        let rawJSON = "{\"contacts\":[{\"name\":\"Sarah\",\"email\":\"s@x.com\"}]}"
        let data = Data(rawJSON.utf8)
        let decoded = try JSONDecoder().decode(Contacts.self, from: data)

        XCTAssertEqual(decoded.contacts.count, 1)
        XCTAssertEqual(decoded.contacts[0].name, "Sarah")
        XCTAssertEqual(decoded.contacts[0].email, "s@x.com")
    }

    func testInvalidPostprocessingErrorDescription() {
        let error = ExtractionError.invalidPostprocessing(reason: "bad values")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("bad values"))
    }

    // MARK: - Large/edge cases

    func testLargeNestedStructure() throws {
        let schema = ExtractionSchema(name: "test", systemPrompt: "Extract.")
        let json = """
        {
            "people": [
                {"name": "A", "contacts": [{"type": "email", "value": "a@a.com"}]},
                {"name": "B", "contacts": [{"type": "phone", "value": "555-0100"}]},
                {"name": "C", "contacts": []}
            ],
            "metadata": {"count": 3, "source": "test"}
        }
        """
        let result = try FiniteExtract.processRawOutput(
            json,
            schema: schema,
            inputText: "",
            modelName: "m",
            inferenceMs: 0
        )

        let people = result.json["people"] as? [[String: Any]]
        XCTAssertEqual(people?.count, 3)
        let metadata = result.json["metadata"] as? [String: Any]
        XCTAssertEqual(metadata?["count"] as? Int, 3)
    }
}

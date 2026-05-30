import XCTest
@testable import FiniteExtract

/// Tests for the staged pipeline that don't require a loaded model.
final class PipelineTests: XCTestCase {

    /// A stage that mutates the context via a supplied closure — lets us drive the
    /// pipeline without MLX.
    struct MarkerStage: ExtractionStage {
        let name: String
        let write: @Sendable (inout ExtractionContext) -> Void
        func run(_ context: inout ExtractionContext) async throws { write(&context) }
    }

    func testStagesRunInOrderAndAssembleResult() async throws {
        let pipeline = ExtractionPipeline([
            MarkerStage(name: "a") { ctx in
                ctx.intermediate = "x"
                ctx.userInfo["modelName"] = "test-model"
            },
            MarkerStage(name: "b") { ctx in
                ctx.json = ["ok": true, "seen": ctx.intermediate ?? ""]
            },
        ])

        let result = try await pipeline.run(on: "hello")

        XCTAssertEqual(result.json["ok"] as? Bool, true)
        XCTAssertEqual(result.json["seen"] as? String, "x")
        XCTAssertEqual(result.rawOutput, "x")
        XCTAssertEqual(result.metadata.modelName, "test-model")
        XCTAssertEqual(result.metadata.stages.map(\.name), ["a", "b"])
    }

    func testLexiconTyperTypesUnambiguousAndAbstainsOnCollision() {
        let typer = LexiconEntityTyper([
            "person": ["Sarah", "Tom", "Rex"],   // Rex appears under both labels →
            "pet": ["Whiskers", "Rex"],          // collision → abstain
        ])

        let spans = typer.typedSpans(in: "Met Sarah and her cat Whiskers. Rex barked at Tom.")
        let map = Dictionary(spans.map { ($0.text.lowercased(), $0.label) }, uniquingKeysWith: { a, _ in a })

        XCTAssertEqual(map["sarah"], "person")
        XCTAssertEqual(map["tom"], "person")
        XCTAssertEqual(map["whiskers"], "pet")
        XCTAssertNil(map["rex"], "ambiguous name should be dropped by the collision-aware lexicon")
    }

    func testCompositeTyperFallbackOnlySeesUndecidedNames() async throws {
        struct StubTyper: EntityTyper {
            func typedSpans(in text: String) -> [TypedSpan] { [TypedSpan(text: "Rex", label: "pet")] }
        }

        let composite = CompositeEntityTyper(
            lexicon: LexiconEntityTyper(["person": ["Sarah"]]),
            fallback: StubTyper()
        )

        let spans = try await composite.typedSpans(in: "Sarah walked Rex.")
        let map = Dictionary(spans.map { ($0.text.lowercased(), $0.label) }, uniquingKeysWith: { a, _ in a })

        XCTAssertEqual(map["sarah"], "person", "decided by the lexicon fast path")
        XCTAssertEqual(map["rex"], "pet", "deferred to the fallback typer")
    }

    func testJSONRepairStagePopulatesContextJSON() async throws {
        var ctx = ExtractionContext(inputText: "irrelevant")
        ctx.intermediate = "```json\n{\"a\": 1,}\n```"   // fenced + trailing comma
        try await JSONRepairStage().run(&ctx)
        XCTAssertEqual(ctx.json["a"] as? Int, 1)
    }
}

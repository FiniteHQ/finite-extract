import Foundation
import MLXLMCommon
import XCTest
@testable import FiniteExtract

/// Generation and cancellation against a real MLX model.
///
/// Skipped unless `FINITE_EXTRACT_TEST_MODEL_DIR` names a local MLX model directory. MLX needs
/// its Metal shaders, which `swift build` doesn't compile, so run these with xcodebuild:
///
/// ```sh
/// TEST_RUNNER_FINITE_EXTRACT_TEST_MODEL_DIR=/path/to/model \
///   xcodebuild test -scheme FiniteExtract-Package -destination 'platform=macOS' \
///   -only-testing:FiniteExtractTests/ModelGenerationTests
/// ```
///
/// Set `FINITE_EXTRACT_BENCH=1` as well (`TEST_RUNNER_FINITE_EXTRACT_BENCH=1` under xcodebuild)
/// to also run the prefill chunk-size benchmark.
final class ModelGenerationTests: XCTestCase {

    // MARK: - Same output as 0.1.1

    /// 0.1.1 generated text with `ChatSession.respond`, which prefills in 512-token chunks.
    /// With the same split, the cancellable path gives exactly the same text.
    ///
    /// Other chunk sizes can change the text: models with a recurrent state (Qwen3.5's linear
    /// attention layers) round that state to the model's precision at the end of every chunk.
    /// The benchmark below reports which sizes match.
    func testOutputMatchesTheChatSessionPathAtTheSameChunkSplit() async throws {
        let extractor = try await Self.model()
        let cases: [(name: String, note: String)] = [
            ("short", Self.note(paragraphs: 1)),
            ("medium", Self.note(paragraphs: 8)),
            ("long", Self.note(paragraphs: Self.longParagraphs)),
        ]

        for (name, note) in cases {
            let reference = try await Self.chatSessionText(extractor, userPrompt: note, maxTokens: 384)
            let (text, _) = try await extractor.generateText(
                systemPrompt: Self.systemPrompt, userPrompt: note, maxTokens: 384, prefillChunkSize: 512
            )
            try Self.saveForComparison(name: name, reference: reference, chunked: text)

            XCTAssertFalse(reference.isEmpty, name)
            XCTAssertEqual(text, reference, "\(name): output differs from the ChatSession path")
        }
    }

    // MARK: - Cancellation

    func testCancelDuringPrefillStopsWithinOneChunk() async throws {
        let extractor = try await Self.model()
        let chunk = FiniteExtract.defaultPrefillChunkSize
        let recorder = EventRecorder(cancelOn: .prefillChunk(index: 3, tokens: chunk))

        let outcome = await Self.outcome {
            try await extractor.generateText(
                systemPrompt: Self.systemPrompt, userPrompt: Self.note(paragraphs: Self.longParagraphs),
                parameters: FiniteExtract.parameters(maxTokens: 64, temperature: 0),
                prefillChunkSize: chunk, probe: recorder.probe()
            )
        }

        XCTAssertEqual(outcome, "cancelled")
        XCTAssertLessThanOrEqual(recorder.submissionsAfterCancel, 1)
        XCTAssertFalse(recorder.events.contains { if case .decodeStart = $0 { true } else { false } })
    }

    func testCancelDuringGenerationStopsBeforeTheNextStep() async throws {
        let extractor = try await Self.model()
        let recorder = EventRecorder(cancelOn: .token(index: 5))

        let outcome = await Self.outcome {
            try await extractor.generateText(
                systemPrompt: Self.systemPrompt, userPrompt: Self.note(paragraphs: 2),
                parameters: FiniteExtract.parameters(maxTokens: 64, temperature: 0),
                prefillChunkSize: FiniteExtract.defaultPrefillChunkSize, probe: recorder.probe()
            )
        }

        XCTAssertEqual(outcome, "cancelled")
        XCTAssertEqual(recorder.submissionsAfterCancel, 0)
        XCTAssertEqual(recorder.events.last, .token(index: 5))
    }

    // MARK: - Benchmark

    /// Prefill time per chunk, and total prefill time against 512-token chunks (the
    /// mlx-swift-lm default that 0.1.1 used), on a ~2,000-token prompt.
    func testPrefillChunkSizeBenchmark() async throws {
        guard ProcessInfo.processInfo.environment["FINITE_EXTRACT_BENCH"] == "1" else {
            throw XCTSkip("set FINITE_EXTRACT_BENCH=1 to run the benchmark")
        }
        let extractor = try await Self.model()
        let note = Self.note(paragraphs: Self.longParagraphs)
        let sizes = [16, 24, 32, 48, 64, 128, 256, 512]
        let rounds = 5

        // Warm up kernels and caches.
        _ = try await Self.timedPrefill(extractor, note: note, chunkSize: 512)

        var results: [Int: [PrefillTiming]] = [:]
        for _ in 0 ..< rounds {
            for size in sizes {
                results[size, default: []].append(try await Self.timedPrefill(extractor, note: note, chunkSize: size))
            }
        }

        let baseline = Self.median(results[512, default: []].map(\.totalMs))
        var lines = ["prompt tokens: \(results[512]?.first?.promptTokens ?? 0), rounds: \(rounds) (medians)"]
        for size in sizes {
            let timings = results[size, default: []]
            let total = Self.median(timings.map(\.totalMs))
            let perChunk = Self.median(timings.flatMap(\.chunkMs))
            let worst = timings.flatMap(\.chunkMs).max() ?? 0
            let lastUnit = Self.median(timings.map(\.lastUnitMs))
            let overhead = (total / baseline - 1) * 100
            lines.append(String(
                format: "chunk %4d: %3d chunks, per chunk median %6.1f ms (max %6.1f), last unit %6.1f ms, "
                    + "prefill %7.1f ms, %+5.1f%% vs 512",
                size, timings.first?.chunkMs.count ?? 0, perChunk, worst, lastUnit, total, overhead
            ))
        }
        lines.append(String(format: "decode step median: %.1f ms", try await Self.decodeStepMs(extractor)))

        // Same text as the 0.1.1 path at each chunk size?
        for (name, note) in [("medium", Self.note(paragraphs: 8)), ("long", note)] {
            let reference = try await Self.chatSessionText(extractor, userPrompt: note, maxTokens: 384)
            let again = try await Self.chatSessionText(extractor, userPrompt: note, maxTokens: 384)
            lines.append("\(name): reference repeatable: \(reference == again)")
            for size in sizes {
                let (text, _) = try await extractor.generateText(
                    systemPrompt: Self.systemPrompt, userPrompt: note, maxTokens: 384, prefillChunkSize: size
                )
                let firstDifference = zip(text, reference).prefix { $0 == $1 }.count
                lines.append(text == reference
                    ? "\(name) chunk \(size): identical"
                    : "\(name) chunk \(size): differs from character \(firstDifference)")
            }
        }

        let report = lines.joined(separator: "\n")
        print(report)
        try Self.saveReport(report)
    }

    // MARK: - Helpers

    static let systemPrompt = """
        You extract facts from a neighbour's notes. Return ONLY valid JSON of the form \
        {"people": [{"name": "...", "house": "...", "pets": ["..."], "notes": "..."}]}. \
        Use only facts stated in the note.
        """

    /// Paragraphs in the long note: with the system prompt, about 2,000 tokens.
    static let longParagraphs = 30

    /// A note with `paragraphs` paragraphs about made-up neighbours (about 63 tokens each).
    static func note(paragraphs: Int) -> String {
        let first = ["Marta", "Oskar", "Lena", "Tobias", "Ines", "Rafael", "Greta", "Milo", "Sanna", "Hugo"]
        let last = ["Lindqvist", "Okafor", "Brandt", "Moreau", "Kowalczyk", "Haddad", "Novak", "Ferreira"]
        let pets = [
            "a beagle called Pepper", "two cats, Miso and Tofu", "a grey rabbit named Clover",
            "an old labrador called Biscuit", "a parrot that says hello", "no pets",
        ]
        let facts = [
            "is happy for the kids to use the swing", "works nights at the hospital",
            "is learning the cello", "lends out the ladder if you ask", "grows tomatoes by the fence",
            "just had a baby girl", "coaches the under-nines on Saturdays",
        ]
        return (0 ..< paragraphs).map { i in
            let name = "\(first[i % first.count]) \(last[(i * 3) % last.count])"
            let partner = first[(i + 4) % first.count]
            return """
                \(name) lives at number \(10 + i * 2) on Alder Lane with \(partner). They have \
                \(pets[i % pets.count]). \(name.split(separator: " ")[0]) \(facts[i % facts.count]), \
                and \(partner) \(facts[(i + 3) % facts.count]). Their house is the one with the \
                \(["blue", "yellow", "green", "red"][i % 4]) door, two down from the corner shop.
                """
        }.joined(separator: "\n\n")
    }

    /// The 0.1.1 implementation of `generateText`.
    static func chatSessionText(_ extractor: FiniteExtract, userPrompt: String, maxTokens: Int) async throws -> String {
        var parameters = GenerateParameters()
        parameters.maxTokens = maxTokens
        parameters.temperature = 0
        let session = ChatSession(extractor.container, instructions: systemPrompt, generateParameters: parameters)
        return try await session.respond(to: userPrompt)
    }

    struct PrefillTiming {
        var promptTokens = 0
        var chunkMs: [Double] = []
        /// The remaining prompt tokens plus the first sampled token, run as one unit.
        var lastUnitMs = 0.0
        var totalMs = 0.0
    }

    /// Prefill the note once and time each chunk (prefill is synchronous per chunk, so the gap
    /// between one chunk starting and the next is that chunk's time). Total runs from the call
    /// to the first token coming back.
    static func timedPrefill(_ extractor: FiniteExtract, note: String, chunkSize: Int) async throws -> PrefillTiming {
        let clock = TimedEvents()
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await extractor.generateText(
            systemPrompt: systemPrompt, userPrompt: note,
            parameters: FiniteExtract.parameters(maxTokens: 1, temperature: 0),
            prefillChunkSize: chunkSize, probe: clock.probe()
        )
        let events = clock.events
        var timing = PrefillTiming()
        let marks = events.filter {
            switch $0.event {
            case .prefillChunk, .decodeStart: true
            default: false
            }
        }
        for (current, next) in zip(marks, marks.dropFirst()) {
            if case .prefillChunk(_, let tokens) = current.event {
                timing.promptTokens += tokens
                timing.chunkMs.append((next.time - current.time) * 1000)
            }
        }
        if case .decodeStart(let tokens)? = marks.last?.event {
            timing.promptTokens += tokens
        }
        if let firstToken = events.first(where: { $0.event == .token(index: 0) }) {
            timing.totalMs = (firstToken.time - start) * 1000
            if let last = marks.last {
                timing.lastUnitMs = (firstToken.time - last.time) * 1000
            }
        }
        return timing
    }

    /// Median time between tokens while generating (one decode step each).
    static func decodeStepMs(_ extractor: FiniteExtract) async throws -> Double {
        let clock = TimedEvents()
        _ = try await extractor.generateText(
            systemPrompt: systemPrompt, userPrompt: note(paragraphs: 2),
            parameters: FiniteExtract.parameters(maxTokens: 64, temperature: 0),
            prefillChunkSize: FiniteExtract.defaultPrefillChunkSize, probe: clock.probe()
        )
        let tokens = clock.events.filter { if case .token = $0.event { true } else { false } }
        return median(zip(tokens, tokens.dropFirst()).map { ($1.time - $0.time) * 1000 })
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        return sorted[sorted.count / 2]
    }

    /// Run `body` in its own task and say how it ended.
    static func outcome(_ body: @escaping @Sendable () async throws -> (text: String, inferenceMs: Int)) async -> String {
        do {
            _ = try await Task { try await body() }.value
            return "returned normally"
        } catch is CancellationError {
            return "cancelled"
        } catch {
            return "threw \(error)"
        }
    }

    /// With `FINITE_EXTRACT_TEST_OUTPUT_DIR` set, write both outputs so they can be compared with `cmp`.
    static func saveForComparison(name: String, reference: String, chunked: String) throws {
        guard let dir = ProcessInfo.processInfo.environment["FINITE_EXTRACT_TEST_OUTPUT_DIR"] else { return }
        let url = URL(fileURLWithPath: dir)
        try Data(reference.utf8).write(to: url.appendingPathComponent("\(name)-chatsession.txt"))
        try Data(chunked.utf8).write(to: url.appendingPathComponent("\(name)-chunked.txt"))
    }

    static func saveReport(_ report: String) throws {
        guard let dir = ProcessInfo.processInfo.environment["FINITE_EXTRACT_TEST_OUTPUT_DIR"] else { return }
        try Data(report.utf8).write(to: URL(fileURLWithPath: dir).appendingPathComponent("prefill-benchmark.txt"))
    }

    /// The model is loaded once for the whole class.
    static func model() async throws -> FiniteExtract {
        guard let path = ProcessInfo.processInfo.environment["FINITE_EXTRACT_TEST_MODEL_DIR"] else {
            throw XCTSkip("set FINITE_EXTRACT_TEST_MODEL_DIR to a local MLX model directory")
        }
        return try await SharedModel.shared.load(path: path)
    }
}

private actor SharedModel {
    static let shared = SharedModel()
    private var extractor: FiniteExtract?

    func load(path: String) async throws -> FiniteExtract {
        if let extractor { return extractor }
        let loaded = try await FiniteExtract(modelDirectory: URL(fileURLWithPath: path))
        extractor = loaded
        return loaded
    }
}

/// Records probe events with the time each arrived.
private final class TimedEvents: @unchecked Sendable {
    struct Mark {
        let event: GenerationEvent
        let time: CFAbsoluteTime
    }

    private let lock = NSLock()
    private var marks: [Mark] = []

    func probe() -> GenerationProbe {
        GenerationProbe { [self] event in
            let now = CFAbsoluteTimeGetCurrent()
            lock.lock()
            marks.append(Mark(event: event, time: now))
            lock.unlock()
        }
    }

    var events: [Mark] {
        lock.lock()
        defer { lock.unlock() }
        return marks
    }
}

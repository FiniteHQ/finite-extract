import XCTest
@testable import FiniteExtract

/// The cancellation rules of the generation loop, driven by scripted steps instead of a model.
///
/// The rule: once the calling task is cancelled, at most the unit of GPU work already being
/// submitted runs, and the loop throws `CancellationError` with nothing left queued. Prefill
/// chunks finish before the next check, so the loop waits for the GPU only once decoding has
/// started (waiting commits a command buffer, which an app in the background must not do).
final class CancellableGenerationTests: XCTestCase {

    // MARK: - Uncancelled runs

    func testPromptIsSplitLikeMLXPrefill() throws {
        // 200 tokens in chunks of 64: three whole chunks, then the last 8 go to the iterator.
        let steps = ScriptedSteps(promptTokens: 200, tokens: [7, 8, 9])
        var seen: [Int] = []

        try Self.runSteps(steps, chunkSize: 64) { seen.append($0); return true }

        XCTAssertEqual(seen, [7, 8, 9])
        XCTAssertEqual(steps.calls, [
            .prefill(64), .prefill(64), .prefill(64), .startDecoding(8),
            .next, .next, .next, .next, .wait,
        ])
    }

    func testPromptOfExactlyTwoChunksLeavesOneWholeChunkForTheIterator() throws {
        // mlx-swift-lm prefills while more than one chunk is left, so 128 / 64 leaves 64.
        let steps = ScriptedSteps(promptTokens: 128, tokens: [1])

        try Self.runSteps(steps, chunkSize: 64) { _ in true }

        XCTAssertEqual(steps.calls, [.prefill(64), .startDecoding(64), .next, .next, .wait])
    }

    func testShortPromptSkipsPrefillChunks() throws {
        let steps = ScriptedSteps(promptTokens: 20, tokens: [1, 2])

        try Self.runSteps(steps, chunkSize: 64) { _ in true }

        XCTAssertEqual(steps.calls, [.startDecoding(20), .next, .next, .next, .wait])
    }

    func testEndOfSequenceStopsWithoutAskingForMoreTokens() throws {
        let steps = ScriptedSteps(promptTokens: 10, tokens: [1, 2, 3, 4])
        var seen: [Int] = []

        try Self.runSteps(steps, chunkSize: 64) { token in
            seen.append(token)
            return token != 2
        }

        XCTAssertEqual(seen, [1, 2])
        XCTAssertEqual(steps.calls, [.startDecoding(10), .next, .next, .wait])
    }

    func testProbeSeesEveryUnitOfWorkInOrder() throws {
        let steps = ScriptedSteps(promptTokens: 100, tokens: [5, 6])
        let recorder = EventRecorder()

        try Self.runSteps(steps, chunkSize: 64, probe: recorder.probe()) { _ in true }

        XCTAssertEqual(recorder.events, [
            .prefillChunk(index: 0, tokens: 64),
            .decodeStart(tokens: 36),
            .decodeStep(index: 0), .token(index: 0),
            .decodeStep(index: 1), .token(index: 1),
            .decodeStep(index: 2),
        ])
    }

    // MARK: - Cancelling during prefill

    func testCancelledBeforeStartingSubmitsNothing() {
        let steps = ScriptedSteps(promptTokens: 300, tokens: [1])

        assertThrowsCancellation {
            cancelCurrentTask()
            try Self.runSteps(steps, chunkSize: 64) { _ in true }
        }

        XCTAssertEqual(steps.calls, [])
    }

    func testCancelDuringPrefillDoesNotWait() {
        // The chunk has finished when the loop sees the cancel, so there is nothing to wait for.
        let steps = ScriptedSteps(promptTokens: 300, tokens: [1])
        steps.duringPrefill = { _ in cancelCurrentTask() }

        assertThrowsCancellation {
            try Self.runSteps(steps, chunkSize: 64) { _ in true }
        }

        XCTAssertEqual(steps.calls, [.prefill(64)])
    }

    func testCancelWhileAChunkRunsSubmitsNoFurtherChunk() {
        // 640 tokens: nine whole chunks of 64 before the iterator. Cancel while chunk 2 runs.
        let steps = ScriptedSteps(promptTokens: 640, tokens: [1])
        steps.duringPrefill = { index in
            if index == 2 { cancelCurrentTask() }
        }

        assertThrowsCancellation {
            try Self.runSteps(steps, chunkSize: 64) { _ in true }
        }

        XCTAssertEqual(steps.calls, [.prefill(64), .prefill(64), .prefill(64)])
    }

    func testCancelAsAChunkIsSubmittedAllowsAtMostThatChunk() {
        let steps = ScriptedSteps(promptTokens: 640, tokens: [1])
        let recorder = EventRecorder(cancelOn: .prefillChunk(index: 4, tokens: 64))

        assertThrowsCancellation {
            try Self.runSteps(steps, chunkSize: 64, probe: recorder.probe()) { _ in true }
        }

        XCTAssertLessThanOrEqual(recorder.submissionsAfterCancel, 1)
        XCTAssertEqual(steps.calls.filter { $0 == .prefill(64) }.count, 5)
        XCTAssertFalse(steps.calls.contains(.wait))
    }

    func testCancelDuringLastPrefillChunkNeverStartsDecoding() {
        let steps = ScriptedSteps(promptTokens: 150, tokens: [1])
        steps.duringPrefill = { index in
            if index == 1 { cancelCurrentTask() }
        }

        assertThrowsCancellation {
            try Self.runSteps(steps, chunkSize: 64) { _ in true }
        }

        XCTAssertEqual(steps.calls, [.prefill(64), .prefill(64)])
    }

    // MARK: - Cancelling during generation

    func testCancelBetweenTokensStopsBeforeTheNextStep() {
        let steps = ScriptedSteps(promptTokens: 10, tokens: Array(1...20))
        let recorder = EventRecorder(cancelOn: .token(index: 2))

        assertThrowsCancellation {
            try Self.runSteps(steps, chunkSize: 64, probe: recorder.probe()) { _ in true }
        }

        XCTAssertEqual(recorder.submissionsAfterCancel, 0)
        XCTAssertEqual(steps.calls, [.startDecoding(10), .next, .next, .next, .wait])
    }

    func testCancelFromTheTokenHandlerStopsBeforeTheNextStep() {
        let steps = ScriptedSteps(promptTokens: 10, tokens: Array(1...20))

        assertThrowsCancellation {
            try Self.runSteps(steps, chunkSize: 64) { token in
                if token == 4 { cancelCurrentTask() }
                return true
            }
        }

        XCTAssertEqual(steps.calls.filter { $0 == .next }.count, 4)
        XCTAssertEqual(steps.calls.last, .wait)
    }

    func testCancelWhileDecodingStartsAsksForNoToken() {
        let steps = ScriptedSteps(promptTokens: 10, tokens: [1, 2])
        steps.duringStartDecoding = { cancelCurrentTask() }

        assertThrowsCancellation {
            try Self.runSteps(steps, chunkSize: 64) { _ in true }
        }

        XCTAssertEqual(steps.calls, [.startDecoding(10), .wait])
    }

    // MARK: - Errors

    func testErrorWhileDecodingStartsStillWaits() {
        struct Boom: Error {}
        let steps = ScriptedSteps(promptTokens: 100, tokens: [1])
        steps.startDecodingError = Boom()

        XCTAssertThrowsError(try Self.runSteps(steps, chunkSize: 64) { _ in true }) { error in
            XCTAssertTrue(error is Boom)
        }
        XCTAssertEqual(steps.calls, [.prefill(64), .startDecoding(36), .wait])
    }

    // MARK: - Helpers

    private static func runSteps(
        _ steps: ScriptedSteps,
        chunkSize: Int,
        probe: GenerationProbe? = nil,
        onToken: (Int) -> Bool
    ) throws {
        var steps = steps
        try CancellableGeneration.run(&steps, prefillChunkSize: chunkSize, probe: probe, onToken: onToken)
    }

    /// Run `body` in a fresh task (so cancelling it doesn't cancel the test) and assert it
    /// throws `CancellationError`.
    private func assertThrowsCancellation(
        _ body: @escaping @Sendable () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let done = expectation(description: "task finished")
        let outcome = Outcome()
        Task.detached {
            do {
                try body()
                outcome.set("returned normally")
            } catch is CancellationError {
                outcome.set("cancelled")
            } catch {
                outcome.set("threw \(error)")
            }
            done.fulfill()
        }
        wait(for: [done], timeout: 10)
        XCTAssertEqual(outcome.value, "cancelled", file: file, line: line)
    }
}

/// Cancel the task this code is running in, synchronously.
func cancelCurrentTask() {
    withUnsafeCurrentTask { $0?.cancel() }
}

/// Records probe events, optionally cancelling the current task when a given event arrives.
final class EventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [GenerationEvent] = []
    private var cancelIndex: Int?
    private let cancelOn: GenerationEvent?

    init(cancelOn: GenerationEvent? = nil) {
        self.cancelOn = cancelOn
    }

    func probe() -> GenerationProbe {
        GenerationProbe { [self] event in
            lock.lock()
            recorded.append(event)
            let shouldCancel = event == cancelOn && cancelIndex == nil
            if shouldCancel { cancelIndex = recorded.count - 1 }
            lock.unlock()
            if shouldCancel { cancelCurrentTask() }
        }
    }

    var events: [GenerationEvent] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    /// Units of GPU work submitted at or after the event that cancelled the task.
    var submissionsAfterCancel: Int {
        lock.lock()
        defer { lock.unlock() }
        guard let cancelIndex else { return 0 }
        return recorded[cancelIndex...].filter(\.submitsWork).count
    }
}

/// `GenerationSteps` that log each call and hand out scripted tokens.
private final class ScriptedSteps: GenerationSteps, @unchecked Sendable {
    enum Call: Equatable {
        case prefill(Int)
        case startDecoding(Int)
        case next
        case wait
    }

    private(set) var calls: [Call] = []
    private(set) var pendingPromptTokens: Int
    private var tokens: [Int]
    private var prefillCount = 0
    var duringPrefill: ((Int) -> Void)?
    var duringStartDecoding: (() -> Void)?
    var startDecodingError: Error?

    init(promptTokens: Int, tokens: [Int]) {
        self.pendingPromptTokens = promptTokens
        self.tokens = tokens
    }

    func prefill(count: Int) {
        calls.append(.prefill(count))
        duringPrefill?(prefillCount)
        prefillCount += 1
        pendingPromptTokens -= count
    }

    func startDecoding(chunkSize: Int) throws {
        calls.append(.startDecoding(pendingPromptTokens))
        duringStartDecoding?()
        if let startDecodingError { throw startDecodingError }
        pendingPromptTokens = 0
    }

    func nextToken() -> Int? {
        calls.append(.next)
        return tokens.isEmpty ? nil : tokens.removeFirst()
    }

    func waitForSubmittedWork() {
        calls.append(.wait)
    }
}

private final class Outcome: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = "not run"

    func set(_ value: String) {
        lock.lock()
        stored = value
        lock.unlock()
    }

    var value: String {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

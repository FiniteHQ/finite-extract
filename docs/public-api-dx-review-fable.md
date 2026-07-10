# Public API & DX review — finite-extract v0.1.0

**Reviewer:** Claude (Fable 5), senior Swift API-design / DX review
**Date:** 2026-07-09
**Scope:** the public surface of the `FiniteExtract` library product as of `main` (v0.1.0), plus the docs, example app, and CLI that form a new adopter's first hour. The review treats the public API as a contract about to ossify: v0.1.0 is tagged, the package is on Swift Package Index, and every signature shipped at v1.0 becomes expensive to change.

**Verdict up front:** the architecture is right — small surface, staged pipeline, Sendable-first, domain-agnostic. But roughly six *signature-level* decisions are baked into today's model landscape (loose `maxTokens:temperature:` params, a frozen-by-convention model enum, a tuple return, a concrete class where a protocol belongs) and each is cheap to fix now and breaking to fix later. Fix those before v1.0; everything else is additive.

---

## 1. Public-surface map

### Inventory (20 public types/protocols, one module)

| Group | Symbol | File | Notes |
|---|---|---|---|
| Core | `FiniteExtract` (final class, `Sendable`) | `FiniteExtract.swift` | 3 async inits, `extract` ×2, `generateText`, `modelName` |
| Core | `ExtractionSchema` (`@unchecked Sendable`) | `SchemaTemplate.swift` | name + systemPrompt + prompt-builder + postprocess closures |
| Core | `ExtractionResult` (`@unchecked Sendable`) | `ExtractionResult.swift` | `json: [String: Any]`, `rawJSON`, `rawOutput`, `metadata` — **no public init** |
| Core | `TypedExtractionResult<T>` | `ExtractionResult.swift` | Codable-decoded variant — **no public init** |
| Core | `ExtractionMetadata` | `ExtractionResult.swift` | `modelName`, `inferenceTimeMs: Int`, `stages` |
| Core | `ExtractionError` (2 cases) | `FiniteExtract.swift` | `invalidJSON(rawOutput:)`, `invalidPostprocessing(reason:)` |
| Models | `ExtractModel` (String enum, `CaseIterable`) | `ModelRegistry.swift` | 2 Qwen 2.5 cases; `recommended`; exposes `MLXLMCommon.ModelConfiguration` |
| Pipeline | `ExtractionPipeline` | `ExtractionPipeline.swift` | `run(on:)`, `run(on:as:)` |
| Pipeline | `ExtractionStage` (protocol) | `ExtractionPipeline.swift` | `name` + `run(_ context: inout ExtractionContext)` |
| Pipeline | `ExtractionContext` (`@unchecked Sendable`) | `ExtractionPipeline.swift` | `inputText`, `intermediate`, `typedSpans`, `json`, `userInfo`, `trace` |
| Pipeline | `GenerateStage`, `JSONRepairStage`, `TypingStage` | `Stages.swift` | reference stages |
| Pipeline | `TypedSpan`, `StageTrace` | `ExtractionPipeline.swift` | value types |
| Typing | `EntityTyper` (protocol) | `Stages.swift` | one requirement |
| Typing | `LexiconEntityTyper`, `CompositeEntityTyper` | `Stages.swift` | reference implementations |
| Typing | `CoreMLEntityTyper` (+ `Decode` typealias, `TyperError`) | `CoreMLEntityTyper.swift` | throws `decodeNotConfigured` unless caller supplies a `decode` closure |
| Utility | `Postprocessor.extractJSON(from:)` | `Postprocessor.swift` | JSON repair, returns tuple, wraps bare arrays as `{"items": [...]}` |

Products: `FiniteExtract` library + `fe-cli` executable. Dependencies visible to consumers: `mlx-swift-lm` (pinned `.upToNextMinor(from: "2.31.3")`), `swift-docc-plugin`. Tools version 5.9, Swift 5 language mode, platforms macOS 14 / iOS 17.

### Intended primary use case

One async init (downloads and caches an MLX model from Hugging Face), one `ExtractionSchema` (a system prompt), one `extract(from:schema:)` call → JSON. The escape hatch for harder tasks is `ExtractionPipeline` composed of `ExtractionStage`s, with `EntityTyper` for a discriminative typing pre-pass.

### Shortest working integration

```swift
import FiniteExtract

let extractor = try await FiniteExtract()          // ~2 GB first-run download
let schema = ExtractionSchema(name: "contacts", systemPrompt: "…Return ONLY valid JSON: {…}")
let result = try await extractor.extract(from: text, schema: schema)
result.json   // [String: Any]
```

Four lines. **The happy path is genuinely obvious in under 5 minutes from the README** — the hero snippet is minimal, correct, and matches the code. Two caveats keep it from being a clean pass:

1. The README hero prints `result.json` — a `[String: Any]` — so the first impression is a stringly-typed dictionary. The package's *best* DX feature, `extract(from:schema:as: Contacts.self)` returning real Codable types, is absent from the README entirely (it only appears in the customization guide, line 40). Lead with it.
2. The first *runnable* success is muddier than the first *readable* success: the Metal note says `swift build` doesn't compile Metal shaders and to "build and run on a real device or macOS target," but never gives the exact command sequence that works for the CLI (`swift run fe-cli …`? `xcodebuild`? Xcode only?). A new adopter's most likely first move — `swift run fe-cli --prompt … --text …` on their Mac — is undocumented and possibly a trap (see §3).

---

## 2. API-design critique

### 2.1 Naming and ergonomics

**`ExtractionSchema` is not a schema.** It carries no output shape, no field types, no validation — it's a prompt bundle (system prompt + user-prompt builder + postprocess hook). Developers arriving from OpenAI structured outputs / JSON Schema will expect `schema` to constrain the output; here the "schema" lives informally inside the prompt text. Renaming (`ExtractionTask`, `ExtractionPrompt`) is probably not worth the churn now that v0.1.0 is public, but the doc comment should say explicitly: *"the output shape is expressed in your system prompt; the engine does not validate against it."* If a real JSON-schema-constrained mode ever ships (GBNF-style grammar constraints are on the research roadmap), the good name will already be taken — worth one deliberate decision now.

**`ExtractionSchema.name` is required but dead.** `grep` confirms the engine never reads it — not in prompts, not in traces, not in metadata (`SchemaTemplate.swift:42` is the only assignment). A required first parameter that does nothing is API noise and a mild trust cost ("what is this for?"). Give it a default (`name: String = ""`) or wire it into `ExtractionMetadata`/`StageTrace` so it earns its position.

**`Postprocessor` (the type) vs `postprocess:` (the closure) name the same word for two different things.** `Postprocessor.extractJSON` is JSON *repair*; the `postprocess:` closure is caller *fix-up*. A newcomer reading "postprocessing" in the docs can't tell which layer is meant. `JSONRepair.extract(from:)` (or making the type internal — but it's genuinely useful standalone and CONTRIBUTING.md solicits PRs against it, so keep it public) would remove the collision. Renaming a public symbol is breaking → decide now.

**The `{"items": [...]}` array-wrapping is an undocumented magic key.** `Postprocessor.tryParse` (Postprocessor.swift:103) silently wraps top-level JSON arrays under `"items"`. This is observable behavior consumers will code against (a model that returns `[…]` produces `result.json["items"]`), yet it appears only in a doc comment on `extractJSON`, not in the guide or on `extract`. Document it at the `extract` level or expose `public static let arrayWrapperKey = "items"`.

**`GenerateStage`'s two closure inits differ only by closure parameter type.** `init(…, prompt: (ExtractionContext) -> String)` vs `init(…, userPrompt: (String) -> String)` — with trailing-closure syntax the label vanishes and overload resolution silently picks by inference. The doc example in `ExtractionPipeline.swift:93` — `GenerateStage(extractor, systemPrompt: s) { $0 }` — only compiles because `{ $0 }` fails to type-check as `(ExtractionContext) -> String`. This is a footgun: a user who writes `{ ctx in "…\(ctx)" }` gets whichever overload the compiler infers. Distinct labels are not enough with trailing closures; consider making the context-aware form the only closure init and adding a non-closure convenience (`init(_:systemPrompt:userPromptTemplate:)`) or renaming one init's *first* argument.

### 2.2 Parameter-scatter: the biggest quiet ossification

`maxTokens: Int = 2048, temperature: Float = 0.0` appears in **five** public signatures: `extract(from:schema:maxTokens:temperature:)`, `extract(…as:…)`, `generateText(…)`, and both `GenerateStage` inits. The moment the engine needs one more generation knob — `topP`, repetition penalty, stop sequences, a seed, a KV-cache/prefix-reuse flag — all five signatures grow in lockstep, forever, with default-ordering hazards. This is the classic pre-1.0 fix:

```swift
/// Generation-time knobs. New fields are additive (defaulted) and never break call sites.
public struct GenerateOptions: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public init(maxTokens: Int = 2048, temperature: Float = 0.0) { … }
    public static let `default` = GenerateOptions()
}

public func extract(from text: String, schema: ExtractionSchema,
                    options: GenerateOptions = .default) async throws -> ExtractionResult
```

Struct-with-defaulted-init means every future knob is a non-breaking addition. Cost today: trivial. Cost after v1: five deprecated overloads maintained indefinitely.

### 2.3 `ExtractModel` — a public enum over a moving target

`ExtractModel` is a `String` enum whose cases *are* Hugging Face repo paths, plus `CaseIterable`. Three separate stability problems:

1. **Cases can never be removed or renamed** without breaking source (consumers `switch` and iterate `allCases`). Third-party HF repos move, get deprecated, and get superseded — Qwen 2.5 will not be the recommendation in 18 months (the repo already bumped mlx-swift-lm for Qwen 3.5 support). An enum registry of *someone else's* model repos is a treadmill of breaking changes.
2. **`rawValue` couples case identity to a repo path.** If `mlx-community` re-uploads under a new name, the rawValue must change → breaking.
3. **`modelConfiguration` leaks `MLXLMCommon.ModelConfiguration`** into the public surface. That makes any future mlx-swift-lm major-version migration (or a backend swap) a *FiniteExtract* API break. Nothing in-repo consumes this member except `FiniteExtract.init` — it can be internal.

The extensible-registry pattern fixes all three and reads identically at call sites:

```swift
public struct ExtractModel: Sendable, Hashable, Identifiable {
    public let id: String            // HF repo ID (or local marker)
    public let displayName: String
    public let approximateSizeMB: Int

    public init(id: String, displayName: String? = nil, approximateSizeMB: Int = 0) { … }

    public static let qwen2_5_3b   = ExtractModel(id: "mlx-community/Qwen2.5-3B-Instruct-4bit",   displayName: "Qwen 2.5 3B Instruct",   approximateSizeMB: 2000)
    public static let qwen2_5_1_5b = ExtractModel(id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit", displayName: "Qwen 2.5 1.5B Instruct", approximateSizeMB: 1000)
    public static let recommended  = qwen2_5_3b
    public static let known: [ExtractModel] = [qwen2_5_3b, qwen2_5_1_5b]   // replaces CaseIterable
    // internal: var modelConfiguration: ModelConfiguration
}
```

`.recommended` / `.qwen2_5_3b` call sites don't change. Adding models becomes additive; retiring one becomes a deprecation instead of a break; and it subsumes `FiniteExtract(modelID:)` (a custom ID is just `ExtractModel(id:)`), collapsing two inits into one. This also dissolves the SwiftLint `identifier_name` suppression in `ModelRegistry.swift:10`.

### 2.4 The un-mockable core: `GenerateStage` requires the concrete class

`GenerateStage` stores `private let extractor: FiniteExtract` (`Stages.swift:11`). There is no protocol seam over text generation. Consequences:

- **A consumer cannot unit-test any pipeline containing a `GenerateStage`** without instantiating `FiniteExtract` — i.e., without a real Metal device and a ~2 GB model download. The package's *own* test suite proves the point: `PipelineTests` invents a `MarkerStage` to avoid `GenerateStage` entirely, and no test constructs one (`grep GenerateStage Tests/` → zero hits).
- Retrofitting a protocol after v1 is messy: adding `init(_ generator: any TextGenerating, …)` alongside `init(_ extractor: FiniteExtract, …)` creates overload ambiguity at every existing call site once `FiniteExtract` conforms, forcing a deprecation dance.

Fix now, and the tuple problem (§2.5) comes along for free:

```swift
/// The text-generation seam. `FiniteExtract` is the MLX-backed implementation;
/// tests and alternate backends supply their own.
public protocol TextGenerating: Sendable {
    var modelName: String { get }
    func generateText(systemPrompt: String, userPrompt: String,
                      options: GenerateOptions) async throws -> GeneratedText
}

public struct GeneratedText: Sendable {
    public let text: String
    public let inferenceTime: Duration   // or `inferenceMs: Int` to match metadata
    public init(text: String, inferenceTime: Duration) { … }
}

extension FiniteExtract: TextGenerating { … }
// GenerateStage: private let generator: any TextGenerating
```

This single change makes the whole Level-2 story (the differentiating feature of the package) testable, and unlocks a future non-MLX backend without touching `ExtractionPipeline`.

### 2.5 Tuples in public signatures

`generateText` returns `(text: String, inferenceMs: Int)` (`FiniteExtract.swift:127`) and `Postprocessor.extractJSON` returns `(string: String, parsed: [String: Any])?`. Public tuples cannot gain fields, cannot conform to protocols, and produce miserable diffs when they change. Both should be small structs before v1 (`GeneratedText` above; a `RepairedJSON` for the postprocessor if it stays public).

### 2.6 Error surfacing

`ExtractionError` has two cases; three real failure classes fall through the cracks:

1. **Model-load failures leak raw dependency errors.** Network failure, disk-full, missing local weights, HF rate-limits — all surface as whatever `LLMModelFactory`/Hub throws. Consumers can't `catch` a stable type for the single most user-visible failure (first-run download on a phone). Add `case modelLoadFailed(model: String, underlying: Error)` and wrap in the three inits.
2. **`extract(…as:)` throws bare `DecodingError`** with no access to the JSON that failed to decode (`FiniteExtract.swift:177`) — the adopter's #1 debugging need at exactly that moment. Add `case decodingFailed(underlying: Error, rawJSON: String)` (same in `ExtractionPipeline.run(on:as:)`).
3. **`JSONRepairStage` misreports a missing generate stage as bad JSON.** With `context.intermediate == nil` it throws `invalidJSON(rawOutput: "")` (`Stages.swift:78`) — "model output is not valid JSON: " (empty) when the truth is "no generation stage ran before this stage." Add `case missingIntermediate(stageName: String)`.

Note: adding enum cases is source-breaking for consumers who switch exhaustively — one more reason to complete the error taxonomy *before* v1, and to document that `ExtractionError` may gain cases (or mark the switch expectation in docs).

What's already good: `invalidJSON` carries a 200-char prefix of raw output in `errorDescription` — exactly right. `Task.checkCancellation()` before/after generation and between stages is correct and appreciated.

### 2.7 Async / Sendable correctness

Mostly sound, with one real soundness gap and one hygiene item:

- `FiniteExtract: Sendable` is legitimate: `ModelContainer` is declared `Sendable` in mlx-swift-lm, `modelName` is a `let String`. `ChatSession` (non-Sendable) is created per-call and never escapes — correct pattern.
- `ExtractionContext`'s `@unchecked Sendable` is fine (threaded `inout` through one sequential run).
- **`ExtractionResult`'s `@unchecked Sendable` justification is currently false.** The doc comment (`ExtractionResult.swift:5-7`) claims `json` holds only immutable Foundation values "produced by JSONSerialization." But the dict placed into the result is the one *after* the caller's `postprocess` closure ran (`repairAndPostprocess`, `FiniteExtract.swift:228`), and after an `ExtractionPipeline` run it's whatever an arbitrary caller stage put in `context.json`. A postprocess/assemble step can insert an `NSMutableDictionary`/`NSMutableArray` (which passes `isValidJSONObject`), producing a shared mutable reference inside a value claimed Sendable. Cheap, real fix: re-parse from the re-serialized bytes so the invariant the comment claims is actually enforced —

  ```swift
  let reserialized = try JSONSerialization.data(withJSONObject: parsed)
  let immutable = try JSONSerialization.jsonObject(with: reserialized) as! [String: Any]
  return (String(data: reserialized, encoding: .utf8)!, immutable)
  ```

  (and the same in `ExtractionPipeline.run(on:)`, `ExtractionPipeline.swift:118-125`).
- `ExtractionSchema` is marked `@unchecked Sendable` but stores only a `String` and `@Sendable` closures — plain `Sendable` likely compiles; `@unchecked` should be a last resort, not a habit (each one is a thing reviewers must re-justify forever).
- **The package is in Swift 5 language mode** (tools 5.9). All these Sendable claims are hand-audited, not compiler-checked. Adopting Swift 6 strict concurrency *after* v1 risks discovering an API-level concurrency mistake that is then breaking to fix. Bump to tools 6.0 with the language mode set and fix what falls out (one known test-file capture, `receivedText` in `ExtractionPipelineTests`) before tagging v1. This is also a credibility marker for a 2026 Swift package.

### 2.8 The `[String: Any]` core — decide deliberately, then commit

`[String: Any]` appears in `ExtractionResult.json`, both postprocess closure signatures, `ExtractionContext.json`/`userInfo`, and `Postprocessor.extractJSON`. It's the root cause of every `@unchecked Sendable` in the package. The alternative — a `JSONValue` enum (Sendable, Codable, Equatable) — is the "correct" modern design but is a Large change that makes the postprocess-mutation ergonomics genuinely worse (`json["recipes"] = []` becomes enum surgery).

**Recommendation: keep `[String: Any]` for v1, but (a)** apply the §2.7 immutability fix so the Sendable story is sound, **(b)** position `rawJSON` + `extract(as:)`/`run(on:as:)` as the blessed typed path in README and DocC (Codable is the type-safe tree; the dict is a convenience), and **(c)** state in the API docs that the dictionary contains only Foundation value types. That combination captures ~90% of `JSONValue`'s value at ~5% of the cost. What must *not* happen is changing the closure signatures after v1 — that would break every consumer's schema definitions, which is precisely the code they write most of.

### 2.9 Smaller contract risks

- **`ExtractionContext.userInfo["modelName"]` is a stringly side-channel contract.** `GenerateStage` writes it (`Stages.swift:58`); `ExtractionPipeline.run` reads it, defaulting to the sentinel `"pipeline"` (`ExtractionPipeline.swift:129`). A consumer's custom generate stage won't know the key exists, and their `metadata.modelName` silently becomes `"pipeline"`. Promote it: `public var modelName: String?` on `ExtractionContext` (additive), or at minimum `public static let modelNameKey`.
- **`TypedSpan.start`/`end` as two independent `Int?`** allows the incoherent state (`start != nil, end == nil`) and bakes in UTF-16 offsets with no type-level hint. `public let utf16Range: Range<Int>?` is tighter and self-documenting. Changing later is breaking; S now.
- **`ExtractionMetadata.inferenceTimeMs: Int`**: fine, but if you touch it, `Duration` is the idiomatic 2026 choice. Low priority; the `Ms` suffix at least makes it unambiguous.
- **No public inits on `ExtractionResult` / `TypedExtractionResult`** (memberwise inits are internal). Under-exposure: a consumer wrapping the engine behind their own service protocol cannot stub results in *their* tests. Adding `public init` is additive and cheap — but do it deliberately (it slightly widens the "constructed a result that never went through repair" surface, which is fine given the type is a plain data carrier).
- **`CoreMLEntityTyper` ships half-configured and carries PE-domain residue.** Its default state throws `TyperError.decodeNotConfigured`, whose *public error message* cites "§19.4" of a research paper that does not exist in this repo. Its `postFilter` hard-codes pet-domain heuristics — strip "the {breed}", drop breed words, the parameter is even named `breeds` internally while the public label is `descriptorWords` (`CoreMLEntityTyper.swift:55`) — inside an engine whose whole pitch is "we ship mechanisms, you bring the domain." Either (a) make the post-filter a caller-supplied strategy (`postFilter: (@Sendable (TypedSpan) -> TypedSpan?)? = nil` with the current behavior as an opt-in `SpanFilters.properNameHeuristics(descriptorWords:)`), or (b) demote the type to `internal`/`@_spi(Experimental)` until the GLiNER→Core ML conversion lands and it can ship *working*. A public type that can't run in its default configuration is a DX trap and a credibility ding. It's also missing from the DocC Topics list (`Documentation.docc/FiniteExtract.md` — "Entity typing" lists the other three).
- **`LexiconEntityTyper.candidateNames` is Latin-script/unigram only** (regex `\b[A-Z][a-zA-Z]+\b`). Correct for the reference use case and well-commented in code, but the *public* doc comment should state the limitation (no CJK, no lowercase names, single tokens only) so adopters don't file it as a bug.
- **Dependency pinning:** `.upToNextMinor(from: "2.31.3")` on mlx-swift-lm is defensible for a 0.x-adjacent fast-moving dep, but it means any consumer who *also* depends on mlx-swift-lm can hit resolution conflicts with only a minor-version window. Fine for now; revisit at v1 and document the policy in the README. `swift-docc-plugin` as a package dependency adds resolve weight for every consumer — standard practice, acceptable, but worth knowing it's a choice.

### 2.10 Over-/under-exposure summary

**Over-exposed:** `ExtractModel.modelConfiguration` (leaks MLXLMCommon; internal-ize), `CoreMLEntityTyper` in current form (see above), arguably `ExtractionSchema.buildUserPrompt`/`applyPostprocessing` (they exist so stages can call them; harmless, but they're plumbing — fine to keep). **Under-exposed:** a `TextGenerating` protocol (§2.4), public inits on result types, model cache management (below), and `ExtractionContext.modelName`.

**Missing entirely (additive, post-v1 OK but high adopter value):** model lifecycle management. An iOS app shipping this *must* answer "is the model already downloaded?", "how big is it on disk?", "how do I delete it?" — today the HF cache is opaque and the only affordance is `progressHandler`. Sketch: `ExtractModel.isDownloaded: Bool`, `static func removeCachedModel(_ model: ExtractModel) throws`, `var localSizeBytes: Int64?`. Streaming token output is the other obvious post-v1 additive (`extractStream`/`AsyncSequence`), and the current all-at-once design doesn't block it.

---

## 3. DX gaps

1. **README omissions (all S fixes, high leverage):**
   - The typed Codable API (`extract(from:schema:as:)`) — the best five lines in the package — is not in the README. Make it the hero snippet or the second snippet.
   - `Examples/ExtractionDemo` (a complete SwiftUI app with download-progress UX) is never mentioned. `grep -n "Examples" README.md` → nothing.
   - `fe-cli` is never mentioned, yet "try extraction against your own prompt without writing a line of Swift" is the strongest possible first-run story: `swift run fe-cli --prompt "…" --text "…"`.
   - No error-handling paragraph: what `ExtractionError.invalidJSON` means in practice (small models fail sometimes), that retry-once is a reasonable strategy, that `rawOutput` is there for debugging.
   - No runtime-memory guidance. `approximateSizeMB` covers the *download*; adopters deciding whether to ship a 3B model on an iPhone need approximate inference RSS. One table row per model.
2. **The first-run command path is ambiguous (possible trap).** The README's Metal note warns that `swift build` doesn't compile Metal shaders, but never states whether `swift run fe-cli` works on macOS or what the exact working invocation is. If plain SwiftPM execution fails at inference time (historically true for mlx-swift), the *only* CLI the package ships is broken via the obvious command, and the adopter's first ten minutes end in a Metal runtime error. Verify on hardware and write the exact command block into README + CONTRIBUTING. (Unverified here — see final notes.)
3. **Dead references to a private document.** `docs/customization-guide.md:159` tells readers to "see … `docs/research-paper.md` §17" — that file exists only in the private finite-research repo. Same for the §17.6 citation at line 55 and `CoreMLEntityTyper.swift:12`'s "§19.4" (which also appears in a user-facing error string, §2.9). Every one is a broken promise in a public repo. Either publish the paper, or replace the citations with self-contained explanations.
4. **DocC is good but under-curated.** The catalog exists (`Sources/FiniteExtract/Documentation.docc/FiniteExtract.md`) with a decent landing page and Topics — genuinely above-average for a v0.1. Gaps: `CoreMLEntityTyper` missing from Topics; no DocC *article* version of the customization guide (the site should stand alone; today the tutorial lives only as a GitHub-relative markdown file); no small "Handling errors" or "Choosing a model" article.
5. **`fe-cli` rough edges:** `--model banana` silently falls back to the 3B model (`FECli.swift:92` — exact-match `"1.5b"` or default); no `--max-tokens` / `--temperature`; output mixes status lines with JSON on stdout (breaks `fe-cli … | jq`). Print status to stderr, JSON to stdout, and validate `--model` against known IDs (a struct registry from §2.3 gives you the list).
6. **No CHANGELOG and no stated stability policy.** The repo tags releases and sits on SPI, but there's no `CHANGELOG.md` and the README never says what 0.x means ("public API may change until 1.0; we document breaks in the changelog"). For a package whose pitch includes *engine you can build on*, an explicit stability statement is cheap trust.
7. **Context-window truncation guidance is buried.** The one place the ~32K-token silent-truncation risk is documented is a `- Note:` on `extract` (`FiniteExtract.swift:93-95`). This is a data-loss failure mode; it belongs in the README and the guide with a "split long documents" recipe.
8. **What's already strong (keep):** issue templates + PR template + CODEOWNERS + CONTRIBUTING with a real lint policy; the customization guide is well-written with a worked three-stage example; SPI badges and platform matrix; the Metal/simulator warning is prominent; doc comments are consistently substantive (the collision-aware lexicon comment, the `@unchecked` justifications).

---

## 4. Prioritized fixes

Ranked by (adopter impact × cost-to-change-later). Sizes: S < ~1h, M = half-day, L = multi-day — scoped for a Sonnet agent working alone. **P0 items are breaking-if-deferred; do them before v1.0 and ship as v0.2.0 with a migration note (0.x SemVer permits it).**

### P0 — before the API ossifies (all source-breaking to do later)

1. **[M] Introduce `GenerateOptions` and collapse the parameter scatter** (§2.2). Add the struct; change `FiniteExtract.extract` ×2, `generateText`, and both `GenerateStage` inits to take `options: GenerateOptions = .default`. Delete the loose `maxTokens:temperature:` params (0.x break) or keep one deprecated overload on `extract` only. Update guide + README snippets + fe-cli.
2. **[M] Add the `TextGenerating` protocol seam and `GeneratedText` struct; retire the tuple return** (§2.4, §2.5). `FiniteExtract: TextGenerating`; `GenerateStage` stores `any TextGenerating`. Add a `PipelineTests` case exercising `GenerateStage` against a stub generator — the test that is impossible today is the proof the fix worked.
3. **[M] Convert `ExtractModel` from enum to struct registry** (§2.3). Static members keep call sites source-compatible except `CaseIterable`/`switch` users (replace with `ExtractModel.known`). Make `modelConfiguration` internal. Fold `FiniteExtract(modelID:)` into `FiniteExtract(model: ExtractModel(id:))` and deprecate the `modelID:` init. Removes the SwiftLint suppression.
4. **[S] Complete the `ExtractionError` taxonomy** (§2.6): add `modelLoadFailed(model:underlying:)` (wrap all three inits), `decodingFailed(underlying:rawJSON:)` (both typed-decode paths), `missingIntermediate(stageName:)` (JSONRepairStage). Write `errorDescription` for each; no doc references to private papers in error strings.
5. **[S] Fix the `ExtractionResult` Sendable soundness gap** (§2.7): re-parse from re-serialized bytes in `FiniteExtract.repairAndPostprocess` and `ExtractionPipeline.run(on:)` so `json` provably holds only immutable Foundation values. Add a regression test that a postprocess inserting an `NSMutableArray` comes out immutable.
6. **[S] Promote the modelName side-channel**: `public internal(set) var modelName: String?` on `ExtractionContext` (or public var); `GenerateStage` sets it; `ExtractionPipeline.run` reads it (keep the `userInfo` key read as fallback for one release). Kills the `"pipeline"` sentinel for custom stages that set it.
7. **[S] `TypedSpan`: replace `start: Int?`/`end: Int?` with `utf16Range: Range<Int>?`** (§2.9). Update `LexiconEntityTyper`, `CoreMLEntityTyper.postFilter`, tests, and the guide.
8. **[S] Default `ExtractionSchema.name`** (`name: String = ""`) or wire it into metadata; document that the "schema" does not validate output shape (§2.1).
9. **[M] De-domain `CoreMLEntityTyper` or demote it** (§2.9). Preferred: `postFilter` becomes a caller-supplied `@Sendable (TypedSpan) -> TypedSpan?` with the current heuristic offered as `SpanFilters.properNameHeuristic(descriptorWords:)`; error message rewritten self-contained; add to DocC Topics. If the GLiNER conversion won't land before v1, make the type internal instead — don't ship a public type whose default config only throws.
10. **[M] Swift 6 language mode** (§2.7): bump tools-version to 6.0, set `swiftLanguageModes: [.v6]`, fix fallout (known: `receivedText` capture in `ExtractionPipelineTests`; try removing `@unchecked` from `ExtractionSchema` while there). Compiler-checked Sendable before the API freezes.
11. **[S] Decide and document the `Postprocessor` question** (§2.1): rename to `JSONRepair` now or commit to the name forever; either way, document (or symbolize) the `"items"` array-wrap key, and convert the tuple return to a small struct if it stays public.

### P1 — high-leverage, non-breaking, ship alongside

12. **[S] README upgrade** (§3.1): typed `extract(as:)` as hero or second snippet; sections for `fe-cli` ("try it in one command") and `Examples/ExtractionDemo`; error-handling paragraph; context-window/truncation warning promoted from the doc comment; per-model download + approximate runtime-RSS table; a one-line 0.x stability policy.
13. **[S] Kill the dead research-paper references** (§3.3) in `customization-guide.md` (lines 55, 159) and `CoreMLEntityTyper.swift` — replace with self-contained prose or publish the paper.
14. **[S] Public memberwise inits on `ExtractionResult`, `TypedExtractionResult`, `StageTrace`** so consumers can stub results in their own tests (§2.9).
15. **[S] Verify and document the exact first-run command path** (§3.2): does `swift run fe-cli` execute MLX inference on macOS? Write the working invocation(s) into README and CONTRIBUTING; if SwiftPM-run fails, say so explicitly and give the `xcodebuild` alternative.
16. **[M] DocC curation** (§3.4): add `CoreMLEntityTyper` (or remove if internal-ized); port the customization guide into the `.docc` catalog as articles (Getting Started / Custom Pipelines / Entity Typing / Handling Errors); link SPI-hosted docs from the README badges row.
17. **[S] `fe-cli` hardening** (§3.5): validate `--model` against the registry (error, don't silently default), add `--max-tokens`/`--temperature`, route status output to stderr so stdout is clean JSON.
18. **[S] Add `CHANGELOG.md`** (Keep-a-Changelog format), backfill 0.1.0, and record every P0 break in an 0.2.0 entry with migration lines.

### P2 — post-v1 additive (design now, ship later)

19. **[M] Model lifecycle API** (§2.10): `ExtractModel.isDownloaded`, `localSizeBytes`, `removeCachedModel` — the missing piece for real app download-management UX.
20. **[L] Streaming generation** (`AsyncThrowingStream<String, Error>` token stream on `TextGenerating` + an `extractStream` convenience) — additive once the protocol from fix #2 exists.
21. **[L] Grammar-constrained output** (GBNF/JSON-schema enforcement) — if/when it lands, it is the thing that finally earns the name `ExtractionSchema`.
22. **[M] `JSONValue` typed tree** — only if post-v1 demand shows the Codable path isn't enough; per §2.8 the recommendation is *not* to do this for v1.

---

## Appendix: what this review deliberately did not flag

- `@unchecked Sendable` on `ExtractionContext` and (post-fix) `ExtractionResult` — justified and documented; the pattern is fine once the §2.7 invariant is enforced.
- The two-product package layout, platform floor (macOS 14/iOS 17), MIT + NOTICE, CI/lint posture — all appropriate.
- `Postprocessor.extractJSON`'s repair-ladder implementation — readable, well-tested, and the right scope for public contribution (per CONTRIBUTING).
- The `ExtractionStage`/`inout ExtractionContext` design — unusual (vs. functional `(Input) -> Output` stages) but a good fit for the accumulate-and-trace model, and it's already proven by the private PE port.

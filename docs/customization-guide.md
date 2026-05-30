# Customizing finite-extract for your use case

finite-extract is a generic on-device extraction engine. It ships *mechanisms* — model
inference, JSON repair, a staged pipeline, an entity-typing pre-pass — and you supply the
*domain*: your prompts, your output shape, and (optionally) your own typing model and
post-processor. The library bundles **no** schemas, prompts, lexicons, or domain models. You
bring those.

There are two levels of use. Start at level 1; reach for level 2 only when a single LLM call
isn't enough.

---

## Level 1 — A schema and a prompt

For most tasks, define an `ExtractionSchema` (a system prompt, an optional user-prompt builder,
and an optional postprocess hook) and call `extract`:

```swift
let extractor = try await FiniteExtract()   // downloads the default model on first run

let recipes = ExtractionSchema(
    name: "recipes",
    systemPrompt: """
        Extract recipes from text. Return ONLY valid JSON:
        {"recipes": [{"name": "...", "ingredients": ["..."], "minutes": 0}]}
        """,
    userPromptBuilder: { text in "Extract recipes from:\n\n\(text)" },
    postprocess: { json, _ in
        // optional: normalize, default missing fields, drop junk — runs on the parsed dict
        if json["recipes"] == nil { json["recipes"] = [] }
    }
)

let result = try await extractor.extract(from: text, schema: recipes)
print(result.json)                                   // [String: Any]

// Or decode straight into your types:
struct Book: Decodable { let recipes: [Recipe] }
let typed = try await extractor.extract(from: text, schema: recipes, as: Book.self)
```

`extract(from:schema:)` is itself just a two-stage pipeline — generate, then JSON-repair — so
anything you can express as a schema, you get for free. Use a custom model with
`FiniteExtract(modelID: "org/your-mlx-model")`.

**This is all most use cases need.** If your output is a flat-ish JSON shape and one LLM call
gets it right, stop here.

---

## Level 2 — A custom pipeline

Some tasks benefit from splitting work across stages — the principle the engine was built
around (research-paper §17.6):

> **Type entities discriminatively, generate attributes, assemble in code.** Put each guarantee
> where it is structurally cheapest. A small NER model can't hallucinate an entity type the way
> an LLM can; deterministic code can't drift the way model weights can.

Reach for a pipeline when you hit any of these:
- The LLM confuses entity *types* (e.g. a pet named "Charlie" extracted as a person).
- You want the LLM to emit a compact natural-language/DSL intermediate and convert it to your
  schema deterministically (smaller outputs, structural guarantees, no JSON-bracket failures).
- You need source-faithfulness or schema validation the model can't be trusted to enforce.

### The pieces

A pipeline is an ordered list of `ExtractionStage`s, each mutating a shared `ExtractionContext`:

```swift
public struct ExtractionContext {
    let inputText: String         // immutable input
    var typedSpans: [TypedSpan]   // filled by a typing stage
    var intermediate: String?     // filled by a generate stage (NL / DSL / JSON)
    var json: [String: Any]       // filled by an assemble stage — the final result
    var userInfo: [String: Any]   // scratch space for your own stages
}

public protocol ExtractionStage: Sendable {
    var name: String { get }
    func run(_ context: inout ExtractionContext) async throws
}
```

Engine-provided stages:
- **`GenerateStage`** — runs the model. You give it a system prompt and a prompt builder (which
  can read `typedSpans` to annotate the input). Sets `context.intermediate`.
- **`JSONRepairStage`** — the default assembler: repairs the model's JSON and applies your
  postprocess closure. Sets `context.json`.
- **`TypingStage`** — wraps an `EntityTyper` to fill `context.typedSpans`.

You provide the domain-specific stages (typically the assembler) by conforming to
`ExtractionStage`.

### Entity typing (`EntityTyper`)

If your task has type confusion, add a typing pre-pass. Three building blocks ship:

```swift
// 1. Pure lexicon, collision-aware: names you supply per label. A name under >1 label
//    is ambiguous and abstains. No model, no download.
let lexicon = LexiconEntityTyper([
    "person": ["Sarah", "Tom", "Dana"],
    "org":    ["Acme", "Globex"],
])

// 2. A Core ML NER model (e.g. a GLiNER variant) for zero-shot typing. You bring the
//    converted .mlmodelc, the labels, and a `decode` closure for its I/O contract.
let ner = try CoreMLEntityTyper(modelURL: myGliner, labels: ["person", "org"],
                                decode: myDecode)

// 3. Compose them: lexicon decides what it can for free, NER handles the rest.
let typer = CompositeEntityTyper(lexicon: lexicon, fallback: ner)
```

Or conform to `EntityTyper` yourself — it's one method, `typedSpans(in:) async throws ->
[TypedSpan]`. The `label` vocabulary is entirely yours; the engine never assumes a fixed set.

### A worked three-stage pipeline

```swift
struct AssembleInvoice: ExtractionStage {
    let name = "assemble"
    func run(_ ctx: inout ExtractionContext) async throws {
        let lines = MyDSL.parse(ctx.intermediate ?? "")   // your deterministic parser
        ctx.json = ["vendor": MyDSL.vendor(lines, typedAs: ctx.typedSpans),
                    "lineItems": lines.map(\.asJSON),
                    "total": lines.reduce(0) { $0 + $1.amount }]
    }
}

let pipeline = ExtractionPipeline([
    TypingStage(typer),                                          // → typedSpans
    GenerateStage(extractor, systemPrompt: invoiceDSLSystem) {   // → intermediate (your DSL)
        "Annotate and itemize:\n\($0.inputText)"
    },
    AssembleInvoice(),                                           // → json
])

let result = try await pipeline.run(on: documentText)
// result.metadata.stages → per-stage timing; result.json → your assembled output
```

### Notes

- Stages run in order; throwing aborts the run. Keep the input immutable and write to the
  accumulators.
- The final `context.json` must be JSON-serializable (`[String: Any]` of Foundation value
  types) — the pipeline validates this and throws `ExtractionError.invalidPostprocessing`
  otherwise.
- `run(on:as:)` decodes the result into a `Decodable` type, same as the level-1 typed variant.
- Nothing here is bundled for you: models download on first use (you choose which), and
  lexicons/NER models/assemblers are yours to provide. That's the point — the engine is the
  reusable core, your domain is the application layer.

For a full real-world pipeline built this way, see how the research pipeline composes a
lexicon+GLiNER typer, a natural-language intermediate, and a deterministic assembler in
`docs/research-paper.md` §17.

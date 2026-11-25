# finite-extract

[![CI](https://github.com/FiniteHQ/finite-extract/actions/workflows/ci.yml/badge.svg)](https://github.com/FiniteHQ/finite-extract/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/FiniteHQ/finite-extract)](https://github.com/FiniteHQ/finite-extract/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Swift](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FFiniteHQ%2Ffinite-extract%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/FiniteHQ/finite-extract)
[![Platforms](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FFiniteHQ%2Ffinite-extract%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/FiniteHQ/finite-extract)

On-device structured entity extraction for Apple platforms, powered by [MLX](https://github.com/ml-explore/mlx-swift). Turn free-form text into structured JSON using small language models — **no cloud, no API keys, no network** at inference time.

```swift
let extractor = try await FiniteExtract()

let schema = ExtractionSchema(
    name: "contacts",
    systemPrompt: """
        Extract contacts from text. Return ONLY valid JSON:
        {"contacts": [{"name": "...", "email": "...", "company": "..."}]}
        """
)

let result = try await extractor.extract(from: text, schema: schema)
print(result.json)
```

## What it does

finite-extract is a generic extraction **engine**. You bring the domain — a system prompt and an output shape — and it handles the rest: model loading and inference (MLX), JSON repair, and a composable pipeline for tasks that need more than one model call. It bundles no schemas, prompts, or domain models; everything domain-specific is yours to supply.

Two levels of use:

1. **Schema + prompt** — define an `ExtractionSchema`, call `extract`. Most tasks need only this.
2. **Custom pipeline** — for tasks that need a discriminative entity-typing pre-pass, a natural-language/DSL intermediate, or a deterministic assembler, compose an `ExtractionPipeline` from `ExtractionStage`s.

See **[docs/customization-guide.md](docs/customization-guide.md)** for both, with worked examples. API reference is published at **[finitehq.github.io/finite-extract](https://finitehq.github.io/finite-extract/documentation/finiteextract/)**.

## Installation

Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/finitehq/finite-extract", from: "0.1.0"),
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "FiniteExtract", package: "finite-extract")
    ]),
]
```

Requires macOS 14+ / iOS 17+ on Apple Silicon. Models download from Hugging Face on first use and cache locally; you can also load a local model directory with `FiniteExtract(modelDirectory:)`.

> **Metal note:** MLX inference runs on-device only — it does not run on the iOS Simulator, and `swift build` does not compile the Metal shaders. Build and run on a real device or macOS target.

## Cancelling

`extract`, `ExtractionPipeline.run` and `generateText` stop when the task running them is cancelled. From 0.1.2 the model stops within one small piece of GPU work: one chunk of the prompt (`FiniteExtract.defaultPrefillChunkSize`, 32 tokens) or one generated token. The call then waits for that work to finish and throws `CancellationError`, so nothing is left running on the GPU once it returns.

This matters on iOS, where the system refuses GPU work from an app in the background. Cancel the task when the app is about to go inactive, and await it to know the GPU is idle:

```swift
let task = Task { try await extractor.extract(from: note, schema: schema) }

// When the app is about to go inactive:
task.cancel()
_ = await task.result   // returns after the model has stopped
```

To trade stopping speed for prefill speed, pass `prefillChunkSize:` to `generateText`. On an M4 Mac with a 0.8B 4-bit model and a 2,000-token prompt, 32-token chunks take about 17 ms each and add about 24% to prefill time against 512-token chunks; 64-token chunks take about 28 ms and add about 2%. The chunk size can change the generated text slightly, because some models round an internal state at the end of every chunk. `prefillChunkSize: 512` reproduces 0.1.1's output exactly.

## Architecture

```
Text + Schema
     │
     ▼   ExtractionPipeline
 ┌── GenerateStage ──┐   MLX LLM inference (your prompt → model output)
 │   JSONRepairStage │   fence/comma/quote repair → parsed JSON
 └───────────────────┘
     │
     ▼
ExtractionResult { json, rawJSON, rawOutput, metadata }
```

`extract(from:schema:)` is sugar for this two-stage pipeline. Richer pipelines insert a `TypingStage` (an `EntityTyper` — lexicon and/or a Core ML NER model) and swap in a deterministic assemble stage. The design principle: **type entities discriminatively, generate attributes, assemble in code — put each guarantee where it's structurally cheapest.**

## License

MIT — see [LICENSE](LICENSE). Built by [Finite](https://heyfinite.com). Third-party attributions in [NOTICE](NOTICE).

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

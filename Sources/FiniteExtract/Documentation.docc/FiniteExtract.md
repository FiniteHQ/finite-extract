#  ``FiniteExtract``

On-device structured entity extraction for Apple platforms.

## Overview

`FiniteExtract` runs small language models directly on Apple Silicon via [MLX](https://github.com/ml-explore/mlx-swift) and converts free-form text into typed JSON against a schema you provide. No cloud, no API keys, no network at inference time.

Two levels of use:

- **One-shot extraction.** Build an ``ExtractionSchema`` (system prompt + optional postprocessing) and call `extract(from:schema:)` on ``FiniteExtract/FiniteExtract``. Most tasks need only this.
- **Custom pipeline.** When you need a discriminative entity-typing pre-pass, a natural-language intermediate, or a deterministic assembler, compose an ``ExtractionPipeline`` from ``ExtractionStage`` values.

```swift
let extractor = try await FiniteExtract()

let schema = ExtractionSchema(
    name: "contacts",
    systemPrompt: """
        Extract contacts from text. Return ONLY valid JSON:
        {"contacts": [{"name": "...", "email": "..."}]}
        """
)

let result = try await extractor.extract(from: text, schema: schema)
print(result.json)
```

> Important: MLX inference does **not** run on the iOS Simulator. Build for a real Apple Silicon device or macOS target.

For installation, requirements, and a worked customization tutorial, see the [README](https://github.com/FiniteHQ/finite-extract#readme) and the [customization guide](https://github.com/FiniteHQ/finite-extract/blob/main/docs/customization-guide.md).

## Topics

### Essentials

- ``FiniteExtract/FiniteExtract``
- ``ExtractionSchema``
- ``ExtractionResult``
- ``TypedExtractionResult``
- ``ExtractionMetadata``
- ``ExtractionError``

### Pipeline composition

- ``ExtractionPipeline``
- ``ExtractionStage``
- ``ExtractionContext``
- ``GenerateStage``
- ``JSONRepairStage``
- ``TypingStage``
- ``StageTrace``

### Models

- ``ExtractModel``

### Postprocessing

- ``Postprocessor``

### Entity typing

- ``EntityTyper``
- ``CompositeEntityTyper``
- ``LexiconEntityTyper``
- ``TypedSpan``

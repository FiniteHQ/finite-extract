# Contributing to finite-extract

Thanks for your interest in contributing! Here's how to get started.

## Getting Started

1. Fork the repo and clone your fork
2. Open `ios/` in Xcode or use `swift build` from the `ios/` directory
3. Run tests: `swift test` from `ios/`

**Note:** `swift build` compiles Swift but does not compile Metal shaders. For full MLX runtime testing, use `xcodebuild` or run from Xcode.

## What We're Looking For

- **New model support** — Add entries to `ModelRegistry.swift` for models you've tested
- **JSON repair improvements** — Edge cases in `Postprocessor.extractJSON()` that handle more model output quirks
- **Bug fixes** — Anything that makes extraction more reliable
- **Documentation** — Better examples, guides, or API docs

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass (`swift test` from `ios/`)
4. Open a PR with a clear description of what and why

Keep PRs focused. One feature or fix per PR.

## Code Style

- Follow existing Swift conventions in the codebase
- No force unwraps in library code (tests are fine)
- Keep the public API surface small — prefer internal visibility unless there's a clear reason to expose something

## Adding Models

To add a new model to the registry:

1. Test it against the benchmark suite (`benchmark/` directory) to verify extraction quality
2. Add a case to `ExtractModel` in `ModelRegistry.swift`
3. Document the model's accuracy, size, and device requirements
4. Open a PR with benchmark results

## Reporting Issues

- Use GitHub Issues
- Include: what you expected, what happened, model used, and any raw output if relevant

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

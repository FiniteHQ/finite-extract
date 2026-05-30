# Contributing to finite-extract

Thanks for your interest in contributing! Here's how to get started.

## Getting Started

1. Fork the repo and clone your fork
2. `swift build` from the repo root, or open `Package.swift` in Xcode
3. Run tests: `swift test` from the repo root

**Note:** `swift build` compiles Swift but does not compile Metal shaders. For full MLX runtime testing, build for a real device or macOS target via `xcodebuild` or run from Xcode — the iOS Simulator is not supported (see the Metal note in the README).

## What We're Looking For

- **New model support** — Add entries to `ModelRegistry.swift` for models you've tested
- **JSON repair improvements** — Edge cases in `Postprocessor.extractJSON()` that handle more model output quirks
- **Bug fixes** — Anything that makes extraction more reliable
- **Documentation** — Better examples, guides, or API docs

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass (`swift test` from the repo root)
4. Open a PR with a clear description of what and why

Keep PRs focused. One feature or fix per PR.

## Code Style

- Follow existing Swift conventions in the codebase
- No force unwraps in library code (tests are fine)
- Keep the public API surface small — prefer internal visibility unless there's a clear reason to expose something

### Linting

We run [SwiftLint](https://github.com/realm/SwiftLint); CI fails on any violation (`swiftlint lint --strict`). To check locally before pushing:

```sh
brew install swiftlint   # once
swiftlint lint --strict
```

The configuration in `.swiftlint.yml` deliberately keeps all correctness- and complexity-oriented rules enabled and only relaxes a few purely-stylistic ones.

**Inline `// swiftlint:disable` is an escape hatch of last resort.** If you add one:

- Scope it as tightly as possible (`disable:next` for a single line; a `disable`/`enable` pair around the smallest possible block — never a whole file).
- Put a comment right next to it explaining *why the rule genuinely does not apply here*. "It was noisy" is not a reason.
- Expect it to be scrutinized in review. CI lists every suppression in the tree as a warning annotation, and a reviewer will push back on any that look like silencing a real signal rather than a true exception. Prefer fixing the code over disabling the rule.

## Adding Models

To add a new model to the registry:

1. Add a case to `ExtractModel` in `Sources/FiniteExtract/ModelRegistry.swift`
2. Verify it loads and produces valid extractions on a representative sample of inputs in your own use case
3. Document the model's quantization, RSS footprint, and device requirements in the PR description
4. Open a PR with the extraction-quality numbers you measured and the harness you used to measure them

## Reporting Issues

- Use GitHub Issues
- Include: what you expected, what happened, model used, and any raw output if relevant

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

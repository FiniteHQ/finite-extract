// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "FiniteExtract",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "FiniteExtract", targets: ["FiniteExtract"]),
        .executable(name: "fe-cli", targets: ["fe-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm/", .upToNextMinor(from: "2.31.3")),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "FiniteExtract",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "fe-cli",
            dependencies: ["FiniteExtract"]
        ),
        .testTarget(
            name: "FiniteExtractTests",
            dependencies: ["FiniteExtract"]
        ),
    ]
)

// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "ExtractionDemo",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "ExtractionDemo",
            dependencies: [
                .product(name: "FiniteExtract", package: "finite-extract"),
            ]
        ),
    ]
)

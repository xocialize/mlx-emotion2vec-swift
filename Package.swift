// swift-tools-version: 6.2
import PackageDescription

// mlx-emotion2vec-swift — the MLXEngine `speechEmotion` package over emotion2vec+ large.
// A thin conformance layer over the standalone inference engine emotion2vec-mlx-swift (product
// `Emotion2VecMLX`), same pattern as mlx-demucs-swift / mlx-mel-roformer-swift. The engine
// contract (MLXToolKit) is a local-path dep for in-workspace dev; the Emotion2VecMLX core is
// pinned to a tagged release. Categorical-only (the dimensional/audeering model is CC-BY-NC and
// out of engine scope). Module/product is `MLXEmotion2Vec`.
let package = Package(
    name: "mlx-emotion2vec-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXEmotion2Vec", targets: ["MLXEmotion2Vec"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/emotion2vec-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXEmotion2Vec",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                // Package identity derives from the repo URL's last path component.
                .product(name: "Emotion2VecMLX", package: "emotion2vec-mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            // Emotion2VecMLX's `EmotionRecogniser` (MLX) isn't Sendable-audited, so awaiting its
            // nonisolated init back into the `@InferenceActor` trips strict region-isolation. The
            // engine serializes all lifecycle on InferenceActor, so v5 mode keeps that a warning —
            // same lever as the other MLXEngine packages.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXEmotion2VecTests",
            dependencies: [
                "MLXEmotion2Vec",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)

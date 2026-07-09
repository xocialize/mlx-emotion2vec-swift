import Foundation
import MLXToolKit

/// Init-time configuration for `Emotion2VecSpeechEmotionPackage` (C9): which published checkpoint
/// to load. Categorical-only — the dimensional (audeering) model is CC-BY-NC and out of scope.
public struct Emotion2VecConfiguration: PackageConfiguration, ModelStorable {
    /// HuggingFace repo holding `emotion2vec_large.safetensors` (+ config).
    public var repo: String
    /// Explicit weights directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/emotion2vec-plus-large-mlx",
                modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension Emotion2VecConfiguration: WeightSourcing {
    /// What `EmotionRecogniser(weightsDirectory:)` reads (categorical model + its config).
    static let requiredFiles = [
        "emotion2vec_large.safetensors", "emotion2vec_large_config.json",
    ]

    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: repo, matching: Self.requiredFiles)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func has(_ dir: URL) -> Bool {
            Self.requiredFiles.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
        }
        // Explicit local directory first (dev escape hatch), then the ModelStore layout.
        if let dir = modelDirectory, has(dir) { return [] }
        if let dir = ModelStore(root: storeRoot).directory(for: repo), has(dir) { return [] }
        return weightSources
    }

    /// The configuration with a nil `modelDirectory` resolved to the store layout — what `load()`
    /// uses AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> Emotion2VecConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension Emotion2VecConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved checkpoint paths; the prewarmer skips them when absent (first launch).
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return Self.requiredFiles.map { dir.appending(path: $0) }
    }
}

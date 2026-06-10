import Foundation
import MLXToolKit

/// Init-time configuration for `Emotion2VecSpeechEmotionPackage` (C9): which published checkpoint
/// to load. Categorical-only — the dimensional (audeering) model is CC-BY-NC and out of scope.
public struct Emotion2VecConfiguration: PackageConfiguration, ModelStorable {
    /// HuggingFace repo holding `emotion2vec_large.safetensors` (+ config).
    public var repo: String
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/emotion2vec-plus-large-mlx",
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo
    }
}

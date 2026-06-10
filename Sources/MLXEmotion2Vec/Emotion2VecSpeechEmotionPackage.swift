import Foundation
import MLXToolKit
import Hub
import Emotion2VecMLX

/// An MLXEngine `speechEmotion` package over **emotion2vec+ large** — 9-class categorical speech
/// emotion recognition. A thin conformance wrapper over the standalone `Emotion2VecMLX` engine
/// (emotion2vec-mlx-swift); all model logic (Data2Vec 2.0 conv extractor + transformer + 9-class
/// head, 16 kHz preprocessing) lives there.
///
/// Engine-owned lifecycle (C13): the engine constructs from an `Emotion2VecConfiguration`, pages
/// weights in with `load()` (downloads the HF snapshot and builds the `EmotionRecogniser`), drives
/// `run(_:)`, and reclaims with `unload()`. Returns a `SpeechEmotionResponse` (structured text).
///
/// **Categorical-only.** The package loads emotion2vec+ (FunASR MODEL_LICENSE, permissive). The
/// dimensional audeering V/A/D model in the core is CC-BY-NC and is not loaded here.
@InferenceActor
public final class Emotion2VecSpeechEmotionPackage: ModelPackage {
    public typealias Configuration = Emotion2VecConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // emotion2vec+ weights: FunASR MODEL_LICENSE (permissive, allowlisted). Port code: MIT.
            license: LicenseDeclaration(weightLicense: .funasrModel, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/emotion2vec-plus-large-mlx",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // ~309 MB fp16 weights + conv/transformer activations over the clip.
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 1_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                SpeechEmotionContract.descriptor(
                    name: "emotion2vec-classify",
                    summary: "emotion2vec+ speech emotion recognition: 9-class categorical (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var recogniser: EmotionRecogniser?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard recogniser == nil else { return }
        let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
        let dir = try await hub.snapshot(from: Hub.Repo(id: configuration.repo),
                                         matching: ["emotion2vec_large.safetensors",
                                                    "emotion2vec_large_config.json"])
        // Categorical-only: the dimensional (audeering) model is out of engine scope.
        recogniser = try await EmotionRecogniser(
            weightsDirectory: dir,
            config: EmotionRecogniserConfig(models: .categorical))
    }

    public func unload() async {
        recogniser = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let recogniser else { throw PackageError.notLoaded }
        guard request.capability == .speechEmotion,
              let req = request as? SpeechEmotionRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // EmotionRecogniser loads + resamples (to 16 kHz mono) from a file URL, so the bytes
        // round-trip through a temp file.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        try req.audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let result = try await recogniser.classify(audioURL: tmp)
        let cat = result.categorical
        let scores = cat.probabilities.map { EmotionScore(label: $0.key.rawValue, score: $0.value) }
        return SpeechEmotionResponse(label: cat.label.rawValue, confidence: cat.confidence, scores: scores)
    }
}

extension Emotion2VecSpeechEmotionPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(Emotion2VecSpeechEmotionPackage.self)
    }
}

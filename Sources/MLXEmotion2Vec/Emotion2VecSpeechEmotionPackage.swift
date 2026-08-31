import Foundation
import MLX
import MLXToolKit
import Emotion2VecMLX

/// Errors specific to the emotion2vec package boundary.
public enum Emotion2VecError: Error, Equatable {
    /// Weight sources are missing and there is no store root (or resolved directory) to
    /// materialize into.
    case missingWeights(String)
}

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
                // PHYS RE-BASELINED 2026-08-31 (AB-T-0107): direct-load harness, task_vm_info
                // phys_footprint at 50 ms through a 10 s classify, two runs. Post-load floor
                // 0.31 GB; held between runs (weights + MLX cache) 0.97 GB; in-run peak up to
                // 1.11 GB — the old flat 1 GB was honest; this is the 1.14 split of the same
                // envelope. Activation scales with clip length (conv over the clip): the 0.4 GB
                // margin covers utterance-scale inputs, not long-form audio.
                footprints: [QuantFootprint(quant: .fp16,
                                            residentBytes: 1_000_000_000,
                                            peakActivationBytes: 400_000_000)],
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
        // Auto-materialize the missing checkpoint into the engine store (dir-less configs only;
        // explicit directories never touch the network), forwarding progress via
        // WeightDownloadProgress so the engine's PreparationMonitor surfaces `.downloading`.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        if !missing.isEmpty {
            guard let storeRoot else {
                throw Emotion2VecError.missingWeights(
                    "no models root set and sources missing: \(missing.map(\.role).joined(separator: ", "))")
            }
            try await WeightMaterializer.materialize(missing, into: storeRoot)
        }
        try Task.checkCancellation()
        guard let dir = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw Emotion2VecError.missingWeights("unresolved weights directory (no store root)")
        }
        // Categorical-only: the dimensional (audeering) model is out of engine scope.
        recogniser = try await EmotionRecogniser(
            weightsDirectory: dir,
            config: EmotionRecogniserConfig(models: .categorical))
    }

    public func unload() async {
        recogniser = nil
        // Return the memory for real: MLX's buffer cache survives the recogniser and holds
        // the working set otherwise — phys_footprint would not fall and engine.evict /
        // R-MEM-1 could not reclaim (AB-T-0107 hygiene sweep).
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). No mid-run checkpoints: classify() is a single forward pass over
        // one short utterance (one MLX eval) — the CAN-3 sub-second exemption.
        try Task.checkCancellation()
        guard let recogniser else { throw PackageError.notLoaded }
        guard request.capability == .speechEmotion,
              let req = request as? SpeechEmotionRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // EmotionRecogniser loads + resamples (to 16 kHz mono) from a file URL, so the bytes
        // round-trip through a temp file.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        try req.audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let result = try await recogniser.classify(audioURL: tmp)
        let cat = result.categorical
        // Descending by score AT SOURCE — probabilities is a Dictionary, and mapping it
        // raw shipped a non-deterministic order every consumer had to re-sort
        // (AB-T-0107 hygiene sweep).
        let scores = cat.probabilities
            .map { EmotionScore(label: $0.key.rawValue, score: $0.value) }
            .sorted { $0.score > $1.score }
        return SpeechEmotionResponse(label: cat.label.rawValue, confidence: cat.confidence, scores: scores)
    }
}

extension Emotion2VecSpeechEmotionPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(Emotion2VecSpeechEmotionPackage.self)
    }
}

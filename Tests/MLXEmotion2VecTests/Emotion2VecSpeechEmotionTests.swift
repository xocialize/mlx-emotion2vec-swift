import Testing
import Foundation
import MLXToolKit
@testable import MLXEmotion2Vec

/// Offline conformance checks — no weights, no Metal. Live classification + label parity is proven
/// in the `MLXEngine Testing` app (the model needs the GPU and the published checkpoint).
struct Emotion2VecSpeechEmotionTests {

    @Test func manifestIsSpeechEmotionAndPermissive() {
        let m = Emotion2VecSpeechEmotionPackage.manifest
        #expect(m.capabilities == [.speechEmotion])
        #expect(m.license.weightLicense == .funasrModel)
        #expect(m.license.portCodeLicense == .mit)
        #expect(m.provenance.sourceRepo == "mlx-community/emotion2vec-plus-large-mlx")
    }

    @Test func licenseGateAdmitsTheFunasrWeights() {
        // The whole point of the allowlist extension: this package must clear permissiveOnly.
        #expect(LicensePolicy.permissiveOnly.evaluate(Emotion2VecSpeechEmotionPackage.manifest.license) == .admitted)
    }

    @Test func manifestRequirements() {
        let r = Emotion2VecSpeechEmotionPackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.first?.quant == .fp16)
    }

    @Test func surfaceIsTheCanonicalSpeechEmotionDescriptor() {
        let surface = Emotion2VecSpeechEmotionPackage.manifest.surfaces.first
        #expect(surface?.capability == .speechEmotion)
        #expect(surface?.parameters.first?.kind == .audio)
    }

    @Test func registrationConstructs() throws {
        let reg = Emotion2VecSpeechEmotionPackage.registration
        #expect(reg.manifest.capabilities == [.speechEmotion])
        let pkg = try reg.makePackage(Emotion2VecConfiguration())
        #expect(pkg is Emotion2VecSpeechEmotionPackage)
    }

    @Test func configurationDefaultsToPublishedRepo() {
        #expect(Emotion2VecConfiguration().repo == "mlx-community/emotion2vec-plus-large-mlx")
    }

    @Test func configurationCodableExcludesEnvironmentRoot() throws {
        var c = Emotion2VecConfiguration()
        c.modelsRootDirectory = URL(fileURLWithPath: "/tmp/should-not-persist")
        let data = try JSONEncoder().encode(c)
        let back = try JSONDecoder().decode(Emotion2VecConfiguration.self, from: data)
        #expect(back.repo == "mlx-community/emotion2vec-plus-large-mlx")
        #expect(back.modelsRootDirectory == nil)
    }
}

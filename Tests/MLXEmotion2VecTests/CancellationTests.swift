// CancellationTests.swift — emotion2vec+ through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled: the entry checkpoint fires before the
// notLoaded guard or weights, so a stub configuration suffices. CAN-3: run() is one forward pass
// over a short utterance (a single MLX eval in EmotionRecogniser.classify — no denoise loop, no
// chunking), so the sub-second exemption is the honest posture; no do/catch on the run() path
// can launder a CancellationError.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXEmotion2Vec

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = Emotion2VecSpeechEmotionPackage(configuration: Emotion2VecConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: SpeechEmotionRequest(audio: Audio(format: .wav, data: Data(),
                                                       sampleRate: 16_000, channels: 1)))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // speechEmotion is not a long-run capability and the manifest declares no multi-GB
        // activation peak (~1 GB resident, no peakActivationBytes) — short-run envelope.
        XCTAssertFalse(
            CancellationConformance.longRunImplied(by: Emotion2VecSpeechEmotionPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: Emotion2VecSpeechEmotionPackage.manifest,
            posture: .subSecondRuns(
                reason: "one classification forward over a short utterance — a single MLX eval "
                    + "(conv extractor + transformer + 9-class head in "
                    + "EmotionRecogniser.classify); no iterative loop to checkpoint"))
        XCTAssertTrue(report.passed, report.summary)
    }
}

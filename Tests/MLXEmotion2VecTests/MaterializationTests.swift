// MaterializationTests.swift — emotion2vec+ through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. Single categorical checkpoint — one declaration covers the
// package.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXEmotion2Vec

final class MaterializationTests: XCTestCase {

    /// Temp dir holding probe files that make an explicit-dir config read as satisfied.
    private func satisfiedDir() throws -> (dir: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "emotion2vec-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in Emotion2VecConfiguration.requiredFiles {
            FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data([0]))
        }
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - Engine MAT gate

    func testMATGate() throws {
        let (dir, cleanup) = try satisfiedDir()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: Emotion2VecConfiguration(),
            satisfiedConfiguration: Emotion2VecConfiguration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    func testDeclaresSingleMainSource() {
        let sources = Emotion2VecConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["main"])
        XCTAssertEqual(sources[0].repo, "mlx-community/emotion2vec-plus-large-mlx")
        XCTAssertEqual(sources[0].matching, Emotion2VecConfiguration.requiredFiles)
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "emotion2vec-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = Emotion2VecConfiguration()
        // Empty store: the source is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // Populate the expected layout.
        let dir = root.appending(path: cfg.repo)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in Emotion2VecConfiguration.requiredFiles {
            FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data([0]))
        }
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout; an explicit dir always wins.
        XCTAssertEqual(cfg.resolved(storeRoot: root).modelDirectory?.path, dir.path)
        let explicit = Emotion2VecConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.modelDirectory?.path, "/x")
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = Emotion2VecConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(
            cfg.prewarmPaths.map(\.path),
            Emotion2VecConfiguration.requiredFiles.map {
                root.appending(path: "mlx-community/emotion2vec-plus-large-mlx/\($0)").path
            })
    }

    func testCodableRoundTrip() throws {
        let cfg = Emotion2VecConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(Emotion2VecConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertNil(decoded.modelDirectory)   // environment-specific, never encoded
    }
}

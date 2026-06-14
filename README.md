# mlx-emotion2vec-swift

The MLXEngine **`speechEmotion`** package over [emotion2vec+](https://github.com/ddlBoJack/emotion2vec) — 9-class speech emotion recognition on Apple Silicon.

A thin conformance layer that wraps the standalone inference engine
[`emotion2vec-mlx-swift`](https://github.com/xocialize/emotion2vec-mlx-swift) (product
`Emotion2VecMLX`) and exposes it to [`mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift)
as a `ModelPackage`. All model logic — Data2Vec 2.0 conv extractor + transformer + 9-class head,
16 kHz preprocessing — lives in the core; this package maps the canonical
`SpeechEmotionRequest → SpeechEmotionResponse` contract onto it and owns the engine lifecycle.

## Capability

| | |
|---|---|
| Capability | `speechEmotion` |
| Input | speech `Audio` (.wav, any rate/channels — resampled to 16 kHz mono) |
| Output | `SpeechEmotionResponse` — dominant `label` + `confidence` + full `[EmotionScore]` distribution |
| Labels | angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown |

**Categorical-only.** The dimensional (audeering V/A/D) model in the core is CC-BY-NC and out of
engine scope; this package loads emotion2vec+ alone (FunASR MODEL_LICENSE, permissive).

## Weights

`mlx-community/emotion2vec-plus-large-mlx` (fp16), selected via `Emotion2VecConfiguration.repo`.

## Usage

```swift
import MLXServeCore
import MLXEmotion2Vec

let engine = MLXServeEngine()
try await engine.register(Emotion2VecSpeechEmotionPackage.registration, configuration: Emotion2VecConfiguration())

let speech = Audio(format: .wav, data: wavBytes)
let response = try await engine.run(SpeechEmotionRequest(audio: speech)) as! SpeechEmotionResponse
print(response.label, response.confidence)   // e.g. "happy" 0.87
```

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-emotion2vec-swift", from: "0.1.0")`, then import `MLXEmotion2Vec` (the conformant `speechEmotion` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port code MIT; weights FunASR MODEL_LICENSE.

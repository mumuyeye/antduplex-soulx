# MOSS audio backend

The MOSS integration calls the unmodified official `MOSS-Audio-Tokenizer`
implementation. It does not copy tokenizer architecture code into this project.

## Contract

- Input is mono audio. Other sample rates are resampled to 24 kHz.
- One duplex chunk is exactly 160 ms (3,840 samples), producing two 80 ms
  MOSS frames.
- All 32 RVQ layers are retained. `model.quantizer.decode_codes(audio_codes)`
  converts their codes to the official 768-dimensional quantized vectors.
- Offline utterances use `model.encode(..., chunk_duration=0.16)` and only the
  final partial chunk is zero-padded.
- Online duplex processing must keep one `model.streaming(batch_size=1)`
  context alive across consecutive chunks. Opening a new context for every
  chunk resets the causal KV cache and changes the codes.

## Usage

```python
from models.moss_audio_tokenizer import MossAudioTokenizerAdapter

adapter = MossAudioTokenizerAdapter(
    config.model_config.moss_tokenizer_path,
    device="cuda",
    chunk_duration=config.model_config.moss_chunk_duration,
    num_quantizers=config.model_config.moss_num_quantizers,
)

with adapter.start_streaming_session() as session:
    encoded = session.encode_chunk(waveform_160ms, sample_rate=24_000)
    compact_audio_embeddings = encoded.quantized_embeddings  # [1, 2, 768]
```

Set `model_config.audio_backend: moss` for `State_Prediction_Model`, and pass
the compact embeddings as `moss_audio_embeddings`. The model projects them
from 768 to 1,024 and scatters them into positions selected by `audio_masks`.
The adapter is deliberately external to the Lightning module, so its frozen
weights are not added to the optimizer or training checkpoints.

## Offline training data

Training stores the official quantized embeddings instead of running MOSS in
the training process. The input dataset must contain `index`, `sequence`,
`wav` (mono float32 bytes) and `sample_rate`:

```bash
python scripts/preprocess_moss.py \
  --input data/source \
  --output data/moss \
  --moss-model-path /root/duplex/data/openmodels/MOSS-Audio-Tokenizer \
  --devices 3,4 \
  --workers-per-device 1
```

Each worker owns one official MOSS model and writes one Parquet shard. Increase
`--workers-per-device` only when the GPU has enough memory for another model
copy. The output stores float16 `[T, 768]` embeddings as compact binary together
with `moss_num_frames`; the dataset restores the shape before padding. All GLM
`<|audio_N|>` values are changed to `<|audio_0|>` position markers, and
preprocessing fails if their count does not exactly equal the MOSS frame count.

The included Fisher example has no waveform and therefore cannot itself be
preprocessed.

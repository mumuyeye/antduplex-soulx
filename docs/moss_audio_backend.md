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

The current Fisher example data contains only precomputed GLM audio-token IDs,
not waveforms. Producing a real MOSS training dataset is a separate data step.

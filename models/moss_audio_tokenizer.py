"""Official MOSS-Audio-Tokenizer integration for causal 160 ms audio chunks.

This module intentionally delegates model construction, streaming state, audio
encoding, and RVQ decoding to the public implementation shipped with
MOSS-Audio-Tokenizer.  It does not reproduce the tokenizer architecture.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from transformers import AutoModel


@dataclass(frozen=True)
class MossEncoding:
    """MOSS codes and their official all-RVQ quantized representation."""

    audio_codes: torch.Tensor
    audio_codes_lengths: torch.Tensor
    quantized_embeddings: torch.Tensor
    valid_samples: int
    padded_samples: int

    @property
    def num_frames(self) -> int:
        return int(self.audio_codes_lengths[0].item())

    @property
    def num_chunks(self) -> int:
        return self.num_frames // 2


class MossAudioTokenizerAdapter:
    """Load and call the official mono 24 kHz MOSS tokenizer.

    Offline encoding uses the documented ``chunk_duration`` API.  Online
    encoding is exposed through :meth:`start_streaming_session`, which keeps
    the official ``model.streaming()`` context alive across successive calls.
    """

    EXPECTED_SAMPLING_RATE = 24_000
    EXPECTED_DOWNSAMPLE_RATE = 1_920
    EXPECTED_NUM_QUANTIZERS = 32
    EXPECTED_CODEBOOK_SIZE = 1_024
    EXPECTED_EMBED_DIM = 768

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        chunk_duration: float = 0.16,
        num_quantizers: int = 32,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MOSS model path does not exist: {self.model_path}"
            )

        self.device = torch.device(device)
        self.chunk_duration = float(chunk_duration)
        self.num_quantizers = int(num_quantizers)
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self._active_session: MossStreamingSession | None = None
        self._validate_model()

    @property
    def sampling_rate(self) -> int:
        return int(self.model.sampling_rate)

    @property
    def downsample_rate(self) -> int:
        return int(self.model.downsample_rate)

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_duration * self.sampling_rate))

    @property
    def frames_per_chunk(self) -> int:
        return self.chunk_samples // self.downsample_rate

    @torch.inference_mode()
    def encode_utterance(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> MossEncoding:
        """Encode an utterance with the official internal streaming path."""

        prepared = self._prepare_waveform(waveform, sample_rate)
        valid_samples = int(prepared.shape[-1])
        remainder = valid_samples % self.chunk_samples
        if remainder:
            prepared = torch.nn.functional.pad(
                prepared, (0, self.chunk_samples - remainder)
            )

        output = self.model.encode(
            prepared.unsqueeze(0).to(self.device),
            num_quantizers=self.num_quantizers,
            return_dict=True,
            chunk_duration=self.chunk_duration,
        )
        return self._build_encoding(
            output,
            valid_samples=valid_samples,
            padded_samples=int(prepared.shape[-1]),
        )

    def start_streaming_session(self) -> "MossStreamingSession":
        """Start one stateful official MOSS streaming context."""

        if self._active_session is not None and not self._active_session.closed:
            raise RuntimeError(
                "Only one MOSS streaming session may be active per model instance."
            )
        session = MossStreamingSession(self)
        session.start()
        return session

    def _validate_model(self) -> None:
        config = self.model.config
        quantizer_config = config.quantizer_kwargs
        actual = {
            "sampling_rate": self.sampling_rate,
            "downsample_rate": self.downsample_rate,
            "num_quantizers": int(quantizer_config["num_quantizers"]),
            "codebook_size": int(quantizer_config["codebook_size"]),
            "code_dim": int(config.code_dim),
        }
        expected = {
            "sampling_rate": self.EXPECTED_SAMPLING_RATE,
            "downsample_rate": self.EXPECTED_DOWNSAMPLE_RATE,
            "num_quantizers": self.EXPECTED_NUM_QUANTIZERS,
            "codebook_size": self.EXPECTED_CODEBOOK_SIZE,
            "code_dim": self.EXPECTED_EMBED_DIM,
        }
        if actual != expected:
            raise ValueError(
                f"Unsupported MOSS model metadata: {actual}; expected {expected}"
            )
        if self.num_quantizers != self.EXPECTED_NUM_QUANTIZERS:
            raise ValueError(
                "AntDuplex-SoulX uses all 32 official MOSS RVQ layers; "
                f"got num_quantizers={self.num_quantizers}."
            )
        if self.chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive.")
        if self.chunk_duration > float(config.causal_transformer_context_duration):
            raise ValueError("chunk_duration exceeds the MOSS causal context duration.")
        if self.chunk_samples % self.downsample_rate:
            raise ValueError(
                "chunk_duration * sampling_rate must be divisible by downsample_rate."
            )
        if self.frames_per_chunk != 2:
            raise ValueError(
                "AntDuplex-SoulX requires exactly two MOSS frames per chunk; "
                f"got {self.frames_per_chunk}."
            )

    def _prepare_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(
                "MOSS-Audio-Tokenizer v1 requires mono waveform [T] or [1, T]; "
                f"got {tuple(waveform.shape)}."
            )
        if waveform.shape[-1] == 0:
            raise ValueError("waveform must contain at least one sample.")
        waveform = waveform.detach().float()
        if int(sample_rate) != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, int(sample_rate), self.sampling_rate
            )
        return waveform.contiguous()

    def _build_encoding(
        self,
        output: Any,
        *,
        valid_samples: int,
        padded_samples: int,
        expected_frames: int | None = None,
    ) -> MossEncoding:
        if output.audio_codes is None or output.audio_codes_lengths is None:
            raise RuntimeError("Official MOSS encode returned no audio codes.")
        audio_codes = output.audio_codes
        lengths = output.audio_codes_lengths
        if audio_codes.dim() != 3 or tuple(audio_codes.shape[:2]) != (
            self.num_quantizers,
            1,
        ):
            raise RuntimeError(
                "Unexpected MOSS audio_codes shape: "
                f"{tuple(audio_codes.shape)}; expected [32, 1, T]."
            )
        num_frames = int(lengths[0].item())
        if num_frames != audio_codes.shape[-1]:
            audio_codes = audio_codes[..., :num_frames]
        if expected_frames is not None and num_frames != expected_frames:
            raise RuntimeError(
                f"Expected {expected_frames} MOSS frames, got {num_frames}."
            )
        if num_frames % self.frames_per_chunk:
            raise RuntimeError(
                f"MOSS returned {num_frames} frames, which is not aligned to "
                f"{self.frames_per_chunk} frames per chunk."
            )

        quantized = self.model.quantizer.decode_codes(audio_codes)
        quantized = quantized.transpose(1, 2).contiguous()
        if tuple(quantized.shape) != (1, num_frames, self.EXPECTED_EMBED_DIM):
            raise RuntimeError(
                "Unexpected MOSS quantized embedding shape: "
                f"{tuple(quantized.shape)}; expected [1, {num_frames}, 768]."
            )
        return MossEncoding(
            audio_codes=audio_codes,
            audio_codes_lengths=lengths,
            quantized_embeddings=quantized,
            valid_samples=valid_samples,
            padded_samples=padded_samples,
        )


class MossStreamingSession(AbstractContextManager["MossStreamingSession"]):
    """A persistent official MOSS KV-cache session for 160 ms chunks."""

    def __init__(self, adapter: MossAudioTokenizerAdapter) -> None:
        self.adapter = adapter
        self._streaming_context: Any = None
        self.closed = True

    def start(self) -> None:
        if not self.closed:
            raise RuntimeError("MOSS streaming session is already active.")
        active = self.adapter._active_session
        if active is not None and active is not self and not active.closed:
            raise RuntimeError(
                "Only one MOSS streaming session may be active per model instance."
            )
        self._streaming_context = self.adapter.model.streaming(batch_size=1)
        try:
            self._streaming_context.__enter__()
        except Exception:
            self._streaming_context = None
            raise
        self.closed = False
        self.adapter._active_session = self

    @torch.inference_mode()
    def encode_chunk(self, waveform: torch.Tensor, sample_rate: int) -> MossEncoding:
        if self.closed:
            raise RuntimeError("MOSS streaming session is closed.")
        prepared = self.adapter._prepare_waveform(waveform, sample_rate)
        if prepared.shape[-1] != self.adapter.chunk_samples:
            raise ValueError(
                "Streaming chunks must be exactly 160 ms after resampling: "
                f"expected {self.adapter.chunk_samples} samples at "
                f"{self.adapter.sampling_rate} Hz, got {prepared.shape[-1]}."
            )
        output = self.adapter.model.encode(
            prepared.unsqueeze(0).to(self.adapter.device),
            num_quantizers=self.adapter.num_quantizers,
            return_dict=True,
        )
        return self.adapter._build_encoding(
            output,
            valid_samples=self.adapter.chunk_samples,
            padded_samples=self.adapter.chunk_samples,
            expected_frames=self.adapter.frames_per_chunk,
        )

    def reset(self) -> None:
        if self.closed:
            raise RuntimeError("MOSS streaming session is closed.")
        self._exit_streaming_context()
        self.closed = True
        self.start()

    def close(self) -> None:
        if self.closed:
            return
        self._exit_streaming_context()
        self.closed = True
        if self.adapter._active_session is self:
            self.adapter._active_session = None

    def _exit_streaming_context(self) -> None:
        context = self._streaming_context
        self._streaming_context = None
        if context is not None:
            context.__exit__(None, None, None)

    def __enter__(self) -> "MossStreamingSession":
        if self.closed:
            self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

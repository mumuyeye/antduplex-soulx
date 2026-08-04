from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.moss_audio_tokenizer import MossAudioTokenizerAdapter


class FakeQuantizer:
    def decode_codes(self, codes):
        values = codes.float().sum(dim=0).unsqueeze(1)
        return values.expand(-1, 768, -1).contiguous()


class FakeMossModel:
    sampling_rate = 24_000
    downsample_rate = 1_920

    def __init__(self):
        self.config = SimpleNamespace(
            quantizer_kwargs={"num_quantizers": 32, "codebook_size": 1024},
            code_dim=768,
            causal_transformer_context_duration=10.0,
        )
        self.quantizer = FakeQuantizer()
        self._streaming = False
        self._frame_offset = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    @contextmanager
    def streaming(self, batch_size=1):
        assert batch_size == 1
        old_streaming = self._streaming
        old_offset = self._frame_offset
        self._streaming = True
        self._frame_offset = 0
        try:
            yield self
        finally:
            self._streaming = old_streaming
            self._frame_offset = old_offset

    def encode(
        self,
        waveform,
        *,
        num_quantizers,
        return_dict,
        chunk_duration=None,
    ):
        assert num_quantizers == 32
        assert return_dict
        frames = waveform.shape[-1] // self.downsample_rate
        offset = self._frame_offset if self._streaming else 0
        frame_ids = torch.arange(offset, offset + frames, device=waveform.device)
        if self._streaming:
            self._frame_offset += frames
        codes = frame_ids.view(1, 1, -1).expand(32, 1, -1).contiguous()
        return SimpleNamespace(
            audio_codes=codes,
            audio_codes_lengths=torch.tensor([frames], device=waveform.device),
        )


@pytest.fixture
def adapter(tmp_path):
    with patch(
        "models.moss_audio_tokenizer.AutoModel.from_pretrained",
        return_value=FakeMossModel(),
    ):
        yield MossAudioTokenizerAdapter(tmp_path, device="cpu")


def test_offline_matches_one_persistent_streaming_session(adapter):
    waveform = torch.randn(adapter.chunk_samples * 3)
    offline = adapter.encode_utterance(waveform, 24_000)
    with adapter.start_streaming_session() as session:
        chunks = [
            session.encode_chunk(chunk, 24_000).audio_codes
            for chunk in waveform.split(adapter.chunk_samples)
        ]

    assert torch.equal(offline.audio_codes, torch.cat(chunks, dim=-1))
    assert offline.quantized_embeddings.shape == (1, 6, 768)


def test_isolated_sessions_are_not_equivalent_to_continuous_streaming(adapter):
    waveform = torch.randn(adapter.chunk_samples * 2)
    continuous = adapter.encode_utterance(waveform, 24_000).audio_codes
    isolated = []
    for chunk in waveform.split(adapter.chunk_samples):
        with adapter.start_streaming_session() as session:
            isolated.append(session.encode_chunk(chunk, 24_000).audio_codes)

    assert not torch.equal(continuous, torch.cat(isolated, dim=-1))


def test_reset_clears_state_and_session_can_be_reused(adapter):
    chunk = torch.randn(adapter.chunk_samples)
    with adapter.start_streaming_session() as session:
        first = session.encode_chunk(chunk, 24_000).audio_codes
        session.encode_chunk(chunk, 24_000)
        session.reset()
        after_reset = session.encode_chunk(chunk, 24_000).audio_codes
    assert torch.equal(first, after_reset)


def test_input_validation_and_single_active_session(adapter):
    with pytest.raises(ValueError, match="mono"):
        adapter.encode_utterance(torch.zeros(2, 3840), 24_000)
    with adapter.start_streaming_session() as session:
        with pytest.raises(RuntimeError, match="Only one"):
            adapter.start_streaming_session()
        with pytest.raises(ValueError, match="exactly 160 ms"):
            session.encode_chunk(torch.zeros(3839), 24_000)
    with pytest.raises(RuntimeError, match="closed"):
        session.encode_chunk(torch.zeros(3840), 24_000)


def test_streaming_chunk_is_resampled_to_official_rate(adapter):
    with adapter.start_streaming_session() as session:
        result = session.encode_chunk(torch.zeros(2560), 16_000)
    assert result.audio_codes.shape == (32, 1, 2)


def test_offline_pads_only_the_final_partial_chunk(adapter):
    result = adapter.encode_utterance(torch.zeros(4000), 24_000)
    assert result.valid_samples == 4000
    assert result.padded_samples == 7680
    assert result.num_chunks == 2

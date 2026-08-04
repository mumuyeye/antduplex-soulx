"""Opt-in integration test against the unmodified official MOSS checkpoint."""

import os

import pytest
import torch

from models.moss_audio_tokenizer import MossAudioTokenizerAdapter


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MOSS_GPU_TESTS") != "1",
    reason="set RUN_MOSS_GPU_TESTS=1 to load the official checkpoint",
)


def test_official_offline_and_persistent_streaming_codes_match():
    torch.manual_seed(7)
    adapter = MossAudioTokenizerAdapter(
        "/root/duplex/data/openmodels/MOSS-Audio-Tokenizer", device="cuda:0"
    )
    waveform = torch.randn(adapter.chunk_samples * 5)
    offline = adapter.encode_utterance(waveform, 24_000)
    with adapter.start_streaming_session() as session:
        streamed = torch.cat(
            [
                session.encode_chunk(chunk, 24_000).audio_codes
                for chunk in waveform.split(adapter.chunk_samples)
            ],
            dim=-1,
        )
    assert torch.equal(offline.audio_codes, streamed)
    assert offline.quantized_embeddings.shape == (1, 10, 768)

    isolated = []
    for chunk in waveform.split(adapter.chunk_samples):
        with adapter.start_streaming_session() as session:
            isolated.append(session.encode_chunk(chunk, 24_000).audio_codes)
    assert not torch.equal(offline.audio_codes, torch.cat(isolated, dim=-1))

    first_chunk = waveform[: adapter.chunk_samples]
    with adapter.start_streaming_session() as session:
        first = session.encode_chunk(first_chunk, 24_000).audio_codes
        session.encode_chunk(first_chunk, 24_000)
        session.reset()
        after_reset = session.encode_chunk(first_chunk, 24_000).audio_codes
    assert torch.equal(first, after_reset)

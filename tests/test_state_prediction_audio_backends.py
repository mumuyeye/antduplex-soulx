from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.state_prediction_model import State_Prediction_Model


class CaptureLlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs_embeds = None

    def forward(self, *, inputs_embeds, labels):
        self.inputs_embeds = inputs_embeds
        return SimpleNamespace(logits=inputs_embeds, labels=labels)


class FakeGlmTokenizer(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def codebook(self, ids):
        return self.table[ids]


class SliceProjector(nn.Module):
    def forward(self, embeddings):
        return embeddings[..., :4]


def bare_model(backend):
    model = State_Prediction_Model.__new__(State_Prediction_Model)
    nn.Module.__init__(model)
    model.audio_backend = backend
    model.model_config = SimpleNamespace(added_audio_token_start=100)
    model.embed_tokens_func = nn.Embedding(128, 4)
    model.audio_projector = nn.Identity()
    model.llm = CaptureLlm()
    return model


def test_glm_forward_preserves_existing_embedding_merge():
    model = bare_model("glm")
    table = torch.randn(8, 4)
    model.glm_tokenizer = FakeGlmTokenizer(table)
    ids = torch.tensor([[5, 101, 6, 103]])
    mask = torch.tensor([[False, True, False, True]])
    labels = ids.clone()

    expected_text = model.embed_tokens_func(ids.masked_fill(mask, 0))
    audio_ids = ids.clone()
    audio_ids[mask] -= 100
    audio_ids[~mask] = 0
    expected_audio = table[audio_ids]
    expected = torch.where(mask.unsqueeze(-1), expected_audio, expected_text)

    model((ids, mask, labels))
    assert torch.equal(model.llm.inputs_embeds, expected)
    assert torch.equal(ids, torch.tensor([[5, 101, 6, 103]]))


def test_moss_compact_embeddings_are_scattered_at_audio_positions():
    model = bare_model("moss")
    model.audio_projector = SliceProjector()
    ids = torch.tensor([[5, 101, 6, 103], [101, 7, 8, 9]])
    mask = ids >= 100
    labels = ids.clone()
    moss = torch.zeros(2, 2, 768)
    moss[0, 0, :4] = 1.0
    moss[0, 1, :4] = 2.0
    moss[1, 0, :4] = 3.0
    moss[1, 1, :4] = 99.0

    model((ids, mask, labels, moss))
    merged = model.llm.inputs_embeds
    assert torch.equal(merged[0, mask[0]], moss[0, :, :4])
    assert torch.equal(merged[1, mask[1]], moss[1, :1, :4])


def test_moss_requires_enough_768_dimensional_frames():
    model = bare_model("moss")
    model.audio_projector = nn.Linear(768, 4)
    ids = torch.tensor([[101, 102]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least as many"):
        model((ids, mask, ids, torch.zeros(1, 1, 768)))
    with pytest.raises(ValueError, match="dim 768"):
        model((ids, mask, ids, torch.zeros(1, 2, 767)))

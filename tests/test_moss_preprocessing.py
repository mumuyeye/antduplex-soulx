from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from models.state_prediction_data import State_Prediction_Dataset
from scripts.preprocess_moss import (
    AUDIO_TOKEN_PATTERN,
    canonicalize_audio_tokens,
    output_schema,
    waveform_from_row,
    write_rows,
)


class FakeTokenizer:
    def encode(self, sequence):
        return [5, 101, 6, 102, 7, 103]


def bare_dataset(max_token_length=4):
    dataset = State_Prediction_Dataset.__new__(State_Prediction_Dataset)
    dataset.model_config = SimpleNamespace(audio_backend="moss")
    dataset.dataset_config = SimpleNamespace(max_token_length=max_token_length)
    dataset.tokenizer = FakeTokenizer()
    dataset.added_audio_token_start = 100
    dataset.special_token_start = 100
    dataset.asr_eos_token_id = 90
    dataset.user_idle_token_id = 91
    dataset.user_nonidle_token_id = 92
    dataset.user_complete_token_id = 93
    dataset.user_incomplete_token_id = 94
    dataset.user_backchannel_token_id = 95
    dataset.IGNORE_INDEX = -100
    dataset.pad_token_id = 0
    return dataset


def test_audio_tokens_are_canonicalized_without_changing_other_tokens():
    sequence = "a<|audio_42|><|user_idle|><|audio_7|>b"
    result = canonicalize_audio_tokens(sequence)
    assert result == "a<|audio_0|><|user_idle|><|audio_0|>b"
    assert len(AUDIO_TOKEN_PATTERN.findall(result)) == 2


def test_float16_embeddings_round_trip_through_parquet(tmp_path):
    path = tmp_path / "train.parquet"
    embeddings = np.arange(2 * 768, dtype=np.float16).reshape(2, 768)
    schema = output_schema()
    with pq.ParquetWriter(path, schema) as writer:
        rows = [
            {
                "index": "sample",
                "sequence": "<|audio_0|><|audio_0|>",
                "moss_num_frames": 2,
                "moss_audio_embeddings": embeddings.tobytes(),
            }
        ]
        write_rows(writer, rows, schema)

    result = pq.read_table(path).to_pylist()[0]
    restored = np.frombuffer(result["moss_audio_embeddings"], dtype=np.float16).reshape(
        result["moss_num_frames"], 768
    )
    assert restored.shape == (2, 768)
    assert np.array_equal(restored, embeddings)


def test_waveform_bytes_are_read_as_float32_mono():
    waveform = np.arange(16, dtype=np.float32)
    restored, sample_rate = waveform_from_row(
        {"wav": waveform.tobytes(), "sample_rate": 16_000}
    )
    assert sample_rate == 16_000
    assert torch.equal(restored, torch.from_numpy(waveform))


def test_dataset_truncates_embeddings_with_audio_positions():
    dataset = bare_dataset(max_token_length=4)
    dataset.data_list = [
        {
            "index": "sample",
            "sequence": "unused-by-fake-tokenizer",
            "moss_audio_embeddings": torch.arange(3 * 768).reshape(3, 768),
        }
    ]
    sample = dataset[0]
    assert sample["input_id"].tolist() == [5, 101, 6, 102]
    assert sample["moss_audio_embeddings"].shape == (2, 768)


def test_dataset_rejects_frame_mismatch():
    dataset = bare_dataset()
    dataset.data_list = [
        {
            "index": "bad",
            "sequence": "unused-by-fake-tokenizer",
            "moss_audio_embeddings": torch.zeros(2, 768),
        }
    ]
    with pytest.raises(ValueError, match="sequence has 3 audio positions"):
        dataset[0]


def test_collator_pads_moss_embeddings():
    dataset = bare_dataset()
    label_names = [
        "label_text",
        "label_eos",
        "label_user_idle",
        "label_user_nonidle",
        "label_user_complete",
        "label_user_incomplete",
        "label_user_backchannel",
    ]
    samples = []
    for length in (2, 1):
        sample = {
            "index": "x",
            "input_id": torch.ones(length, dtype=torch.long),
            "audio_mask": torch.ones(length, dtype=torch.bool),
            "moss_audio_embeddings": torch.ones(length, 768, dtype=torch.float16),
        }
        sample.update(
            {name: torch.ones(length, dtype=torch.long) for name in label_names}
        )
        samples.append(sample)
    dataset.config = SimpleNamespace(
        train_config=SimpleNamespace(
            enable_switch_loss_rate=False, switch_loss_rate_label=""
        ),
        dataset_config=SimpleNamespace(batch_size=2),
    )

    batch = dataset.collator(samples)
    assert batch["moss_audio_embeddings"].shape == (2, 2, 768)
    assert torch.count_nonzero(batch["moss_audio_embeddings"][1, 1]) == 0

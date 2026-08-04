"""Opt-in real-model tests for serial and parallel MOSS preprocessing."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MOSS_GPU_TESTS") != "1",
    reason="set RUN_MOSS_GPU_TESTS=1 to load the official checkpoint",
)


MODEL_PATH = "/root/duplex/data/openmodels/MOSS-Audio-Tokenizer"


def write_input(path):
    rows = []
    for index, chunks in enumerate((1, 2, 1, 2)):
        rng = np.random.default_rng(index)
        waveform = rng.standard_normal(chunks * 3840).astype(np.float32)
        audio_tokens = "".join(f"<|audio_{frame + 1}|>" for frame in range(chunks * 2))
        rows.append(
            {
                "index": f"sample-{index}",
                "sequence": audio_tokens,
                "wav": waveform.tobytes(),
                "sample_rate": 24_000,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def run_preprocessing(input_path, output_path, devices, workers_per_device):
    subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_moss.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--moss-model-path",
            MODEL_PATH,
            "--devices",
            devices,
            "--workers-per-device",
            str(workers_per_device),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def read_output(path):
    rows = pq.read_table(path).to_pylist()
    return {row["index"]: row for row in rows}


def assert_outputs_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for index in expected:
        assert expected[index]["sequence"] == actual[index]["sequence"]
        assert expected[index]["moss_num_frames"] == actual[index]["moss_num_frames"]
        assert (
            expected[index]["moss_audio_embeddings"]
            == actual[index]["moss_audio_embeddings"]
        )


def test_serial_multi_gpu_and_multi_process_outputs_match(tmp_path):
    input_path = tmp_path / "input.parquet"
    write_input(input_path)

    serial = tmp_path / "serial"
    multi_gpu = tmp_path / "multi_gpu"
    multi_process = tmp_path / "multi_process"
    run_preprocessing(input_path, serial, "3", 1)
    run_preprocessing(input_path, multi_gpu, "3,4", 1)
    run_preprocessing(input_path, multi_process, "4", 2)

    expected = read_output(serial)
    assert_outputs_equal(expected, read_output(multi_gpu))
    assert_outputs_equal(expected, read_output(multi_process))

    from omegaconf import OmegaConf
    import torch

    from config.config import RunConfig
    from models.state_prediction_data import State_Prediction_Dataset
    from models.state_prediction_model import State_Prediction_Model

    config = OmegaConf.structured(RunConfig())
    config.model_config.audio_backend = "moss"
    config.model_config.model_name = (
        "/root/duplex/data/openmodels/SoulX-Duplug-0.6B/" "Qwen3-0.6B-expand_vocab_v2"
    )
    config.model_config.enable_lora = False
    config.dataset_config.train_data_path = str(serial)
    config.dataset_config.split_size = 0.25
    dataset = State_Prediction_Dataset(config, "train")
    batch = dataset.collator([dataset[0]])

    device = torch.device("cuda:3")
    model = State_Prediction_Model(config).to(device).eval()
    tensors = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    with torch.inference_mode():
        output = model(
            (
                tensors["input_ids"],
                tensors["audio_masks"],
                tensors["label_text_ids"],
                tensors["moss_audio_embeddings"],
            )
        )
    assert output.logits.shape[:2] == tensors["input_ids"].shape
    assert not any(name.startswith("moss") for name, _ in model.named_parameters())

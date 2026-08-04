"""Precompute official MOSS quantized embeddings for state-prediction data."""

import argparse
import multiprocessing as mp
import re
from pathlib import Path


AUDIO_TOKEN_PATTERN = re.compile(r"<\|audio_\d+\|>")
OUTPUT_BATCH_SIZE = 32


def canonicalize_audio_tokens(sequence):
    """Replace GLM code identities while preserving their 80 ms positions."""

    return AUDIO_TOKEN_PATTERN.sub("<|audio_0|>", sequence)


def load_input_dataset(input_path):
    from datasets import load_dataset

    path = Path(input_path)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported input file type: {suffix}")

    dataset = load_dataset(str(path))
    if "train" not in dataset:
        raise ValueError(f"Input dataset has no train split: {input_path}")
    return dataset["train"]


def waveform_from_row(row):
    """Read the fixed ``wav`` and ``sample_rate`` preprocessing fields."""

    import numpy as np
    import torch
    import torchaudio

    value = row["wav"]
    sample_rate = int(row["sample_rate"])

    if isinstance(value, dict):
        sample_rate = int(value.get("sampling_rate", sample_rate))
        value = value["array"]
    if isinstance(value, (bytes, bytearray, memoryview)):
        waveform = torch.from_numpy(np.frombuffer(value, dtype=np.float32).copy())
    elif isinstance(value, str):
        waveform, file_sample_rate = torchaudio.load(value)
        if file_sample_rate != sample_rate:
            raise ValueError(
                f"sample_rate={sample_rate} does not match {file_sample_rate} in {value}"
            )
    else:
        waveform = torch.as_tensor(value)

    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {tuple(waveform.shape)}")
    return waveform.float(), sample_rate


def output_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("index", pa.string()),
            ("sequence", pa.string()),
            ("moss_num_frames", pa.int32()),
            ("moss_audio_embeddings", pa.binary()),
        ]
    )


def write_rows(writer, rows, schema):
    if rows:
        import pyarrow as pa

        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def worker_main(
    rank,
    world_size,
    device,
    input_path,
    output_dir,
    model_path,
):
    import pyarrow.parquet as pq
    import torch

    from models.moss_audio_tokenizer import MossAudioTokenizerAdapter

    torch.cuda.set_device(device)
    dataset = load_input_dataset(input_path).shard(
        num_shards=world_size, index=rank, contiguous=True
    )
    adapter = MossAudioTokenizerAdapter(model_path, device=f"cuda:{device}")
    schema = output_schema()
    output_path = Path(output_dir) / f"train-{rank:05d}-of-{world_size:05d}.parquet"

    rows = []
    with pq.ParquetWriter(output_path, schema, compression="zstd") as writer:
        for row in dataset:
            sequence = canonicalize_audio_tokens(row["sequence"])
            expected_frames = len(AUDIO_TOKEN_PATTERN.findall(sequence))
            waveform, sample_rate = waveform_from_row(row)
            encoded = adapter.encode_utterance(waveform, sample_rate)
            embeddings = encoded.quantized_embeddings.squeeze(0).cpu().half()
            if embeddings.shape != (expected_frames, 768):
                raise ValueError(
                    f"{row['index']}: sequence has {expected_frames} audio positions, "
                    f"but MOSS produced {embeddings.shape[0]} frames"
                )
            rows.append(
                {
                    "index": str(row["index"]),
                    "sequence": sequence,
                    "moss_num_frames": embeddings.shape[0],
                    "moss_audio_embeddings": embeddings.numpy().tobytes(),
                }
            )
            if len(rows) == OUTPUT_BATCH_SIZE:
                write_rows(writer, rows, schema)
        write_rows(writer, rows, schema)

    print(
        f"worker {rank}/{world_size} on cuda:{device}: "
        f"wrote {len(dataset)} rows to {output_path}",
        flush=True,
    )


def parse_devices(value):
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("--devices must contain at least one GPU")
    return devices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--moss-model-path", required=True)
    parser.add_argument("--devices", type=parse_devices, required=True)
    parser.add_argument("--workers-per-device", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers_per_device < 1:
        raise ValueError("--workers-per-device must be at least 1")

    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_devices = [
        device for device in args.devices for _ in range(args.workers_per_device)
    ]
    context = mp.get_context("spawn")
    processes = []
    for rank, device in enumerate(worker_devices):
        process = context.Process(
            target=worker_main,
            args=(
                rank,
                len(worker_devices),
                device,
                args.input,
                args.output,
                args.moss_model_path,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    failed = [process.pid for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"MOSS preprocessing workers failed: {failed}")


if __name__ == "__main__":
    main()

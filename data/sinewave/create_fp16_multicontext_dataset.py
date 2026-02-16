#!/usr/bin/env python3
"""Create multi-context sine-wave datasets encoded as float16 bit patterns.

Each context writes:
  data/<output_root>/<context>/train.bin (uint16)
  data/<output_root>/<context>/val.bin   (uint16)
  data/<output_root>/<context>/meta.pkl

The uint16 values are the raw IEEE-754 fp16 bit patterns for sine samples.
Use `--numerical_multicontext_input_format fp16_bits` during training so
model.py decodes them through GPT._fp16bits_to_fp32.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np


def _make_wave(num_points: int, period_scale: float, phase: float, amplitude: float, offset: float) -> np.ndarray:
    x = np.arange(num_points, dtype=np.float32)
    radians = (2.0 * math.pi * period_scale * x / float(num_points)) + phase
    values = offset + amplitude * np.sin(radians, dtype=np.float32)
    return values.astype(np.float32)


def _fp32_to_fp16_bits(values: np.ndarray) -> np.ndarray:
    fp16 = values.astype(np.float16)
    return fp16.view(np.uint16)


def _write_context(output_root: Path, context_name: str, train_bits: np.ndarray, val_bits: np.ndarray, metadata: dict) -> None:
    context_dir = output_root / context_name
    context_dir.mkdir(parents=True, exist_ok=True)

    train_bits.astype(np.uint16).tofile(context_dir / "train.bin")
    val_bits.astype(np.uint16).tofile(context_dir / "val.bin")

    with (context_dir / "meta.pkl").open("wb") as f:
        pickle.dump(metadata, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fp16-bit sinewave datasets for numerical multicontext training.")
    parser.add_argument("--output_root", default="sinewave_fp16", help="Output directory under data/.")
    parser.add_argument("--contexts", type=int, default=8, help="Number of contexts to generate.")
    parser.add_argument("--samples", type=int, default=240_000, help="Total samples per context.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--base_period", type=float, default=1.0, help="Base period scale for context 0.")
    parser.add_argument("--period_step", type=float, default=0.125, help="Incremental period scale per context.")
    parser.add_argument("--base_phase", type=float, default=0.0, help="Base phase offset in radians for context 0.")
    parser.add_argument("--phase_step", type=float, default=0.3926990817, help="Phase offset step (default pi/8).")
    parser.add_argument("--base_amplitude", type=float, default=0.5, help="Base sine amplitude for context 0.")
    parser.add_argument("--amplitude_step", type=float, default=0.1, help="Amplitude increment per context.")
    parser.add_argument("--dc_offset", type=float, default=0.0, help="Optional dc offset added to all contexts.")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / "data" / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    train_n = int(args.samples * args.train_ratio)
    val_n = args.samples - train_n

    print(f"Writing {args.contexts} contexts into: {output_root}")
    print(f"Samples/context: train={train_n}, val={val_n}")

    for idx in range(args.contexts):
        period = args.base_period + idx * args.period_step
        phase = args.base_phase + idx * args.phase_step
        amplitude = args.base_amplitude + idx * args.amplitude_step
        context_name = f"s{idx + 1}"

        wave = _make_wave(
            num_points=args.samples,
            period_scale=period,
            phase=phase,
            amplitude=amplitude,
            offset=args.dc_offset,
        )
        bits = _fp32_to_fp16_bits(wave)

        train_bits = bits[:train_n]
        val_bits = bits[train_n:]

        metadata = {
            "tokenizer": "sinewave_fp16_bits",
            "encoding": "ieee754-fp16-bitpattern-in-uint16",
            "vocab_size": 65536,
            "numerical_multicontext_input_format": "fp16_bits",
            "samples": args.samples,
            "train_ratio": args.train_ratio,
            "period_scale": period,
            "phase_radians": phase,
            "amplitude": amplitude,
            "dc_offset": args.dc_offset,
        }

        _write_context(output_root, context_name, train_bits, val_bits, metadata)
        print(
            f"[{context_name}] period={period:.4f}, phase={phase:.4f}, amplitude={amplitude:.4f}, "
            f"train[min,max]=({train_bits.min()}, {train_bits.max()})"
        )


if __name__ == "__main__":
    main()

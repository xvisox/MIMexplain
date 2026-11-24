from __future__ import annotations

import argparse
from pathlib import Path

from explainer.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESIZED_IMAGE_SIZE,
    DEFAULT_TEXT_LAYERS,
)
from explainer.examples import list_example_names
from explainer.io import render_heatmap_overlay, save_heatmap_overlay
from explainer.metaclip2 import MetaClip2ExplanationPipeline, MetaClip2RunConfig


def parse_args() -> argparse.Namespace:
    example_names = list_example_names()
    parser = argparse.ArgumentParser(
        description="Run the canonical MetaCLIP-2 Grad-ECLIP baseline on a bundled example."
    )
    parser.add_argument(
        "--example",
        choices=example_names,
        default=example_names[0],
        help="Which bundled example to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where explanation artifacts will be written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device.",
    )
    parser.add_argument(
        "--text-layers",
        type=int,
        default=DEFAULT_TEXT_LAYERS,
        help="How many final text layers to decompose for Grad-ECLIP.",
    )
    parser.add_argument(
        "--resized-image-size",
        type=int,
        default=DEFAULT_RESIZED_IMAGE_SIZE,
        help="Square resize/crop size used before running MetaCLIP-2.",
    )
    parser.add_argument(
        "--without-key-sim",
        action="store_true",
        help="Disable the CLS-to-patch key-similarity term in Grad-ECLIP.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="MetaCLIP-2 checkpoint name. Keep this within the MetaCLIP-2 family.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MetaClip2RunConfig(
        model_name=args.model_name,
        device=args.device,
        text_layers=args.text_layers,
        with_key_similarity=not args.without_key_sim,
        resized_image_size=args.resized_image_size,
    )
    pipeline = MetaClip2ExplanationPipeline(config)
    result = pipeline.run_example(args.example)
    overlay = render_heatmap_overlay(result.image, result.explanation)
    overlay_path, array_path = save_heatmap_overlay(
        image=overlay,
        explanation=result.explanation,
        output_dir=args.output_dir,
        tag=args.example,
    )
    print(f"example={args.example}")
    print(f"map_size={result.map_size[0]}x{result.map_size[1]}")
    print(f"overlay={overlay_path}")
    print(f"array={array_path}")

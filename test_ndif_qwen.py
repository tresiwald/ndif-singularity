#!/usr/bin/env python3
"""Smoke-test an NDIF Singularity container serving Qwen3.5 models."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal nnsight trace against a local NDIF service."
    )
    parser.add_argument(
        "--host",
        default="http://localhost:5001",
        help="NDIF API host. Default: http://localhost:5001",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key printed by the container (singularity run ndif-qwen35.sif apikey).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="Deployed model name. Default: Qwen/Qwen3.5-4B",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Find the number of vertical asymptotes in the graph of\n"
            "\\[y = \\frac{(x + 8) (x + 5)^2 (x + 1)^3 x^5 (x - 3)^2}"
            "{(x + 7) (x + 5)^2 (x + 1) x (x - 3)^3 (x - 4)}.\\]"
        ),
        help="Prompt to trace. Defaults to a vertical-asymptote counting problem.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import nnsight
    except ImportError:
        print(
            "Missing dependency: nnsight. Install it with "
            "`.venv/bin/python -m pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    nnsight.CONFIG.API.HOST = args.host
    nnsight.CONFIG.set_default_api_key(args.api_key)

    print(f"Testing NDIF at {args.host} with model {args.model!r}")
    print("The Singularity service should already be running in another shell.")

    try:
        model = nnsight.LanguageModel(args.model)
        with model.trace(args.prompt, remote=True):
            hidden = model.model.layers[0].output[0].save()
    except Exception as exc:
        print(f"NDIF Qwen smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(f"Trace succeeded. Layer-0 hidden shape: {tuple(hidden.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Smoke-test an NDIF Singularity container serving GPT-2."""

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
        default="any-key-works-in-dev-mode",
        help="API key to send to NDIF. Default works when NDIF_DEV_MODE=true.",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face model name deployed in NDIF. Default: gpt2",
    )
    parser.add_argument(
        "--prompt",
        default="Hello world",
        help="Prompt to trace. Default: 'Hello world'",
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
            hidden = model.transformer.h[0].output[0].save()
    except Exception as exc:
        print(f"NDIF GPT-2 smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(f"Trace succeeded. Layer-0 hidden shape: {tuple(hidden.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

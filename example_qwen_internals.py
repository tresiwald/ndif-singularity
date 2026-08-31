#!/usr/bin/env python3
"""Extract internals and layerwise uncertainty features from the Qwen3.5 NDIF deployment.

The container serves Qwen/Qwen3.5-4B and Qwen/Qwen3.5-2B behind the NDIF API.
This script uses nnsight to pull, in a single remote trace:

  * per-layer hidden states (residual stream) and attention outputs
  * layerwise uncertainty features via the logit lens: for every layer, the
    hidden state is projected through the final norm + LM head, and we compute
    predictive entropy, max-probability (confidence), and the top-1/top-2
    probability margin for the last token position.
"""

from __future__ import annotations

import argparse

import torch

DEFAULT_PROMPT = (
    "Find the number of vertical asymptotes in the graph of\n"
    "\\[y = \\frac{(x + 8) (x + 5)^2 (x + 1)^3 x^5 (x - 3)^2}"
    "{(x + 7) (x + 5)^2 (x + 1) x (x - 3)^3 (x - 4)}.\\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://localhost:5001", help="NDIF API host.")
    parser.add_argument("--api-key", required=True, help="API key printed by the container.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        choices=["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B"],
        help="Which deployed model to trace.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to analyze. Defaults to a vertical-asymptote counting problem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import nnsight

    nnsight.CONFIG.API.HOST = args.host
    nnsight.CONFIG.set_default_api_key(args.api_key)

    model = nnsight.LanguageModel(args.model)
    layers = model.model.layers
    n_layers = len(layers)

    hidden_states = []
    attn_outputs = []
    entropies = []
    max_probs = []
    margins = []

    with model.trace(args.prompt, remote=True):
        for layer in layers:
            # Residual stream after this layer: (batch, seq, hidden)
            h = layer.output[0]
            hidden_states.append(h.save())

            # Attention block output for the same layer.
            attn_outputs.append(layer.self_attn.output[0].save())

            # Logit lens: project the intermediate hidden state through the
            # final RMSNorm and the LM head to get a per-layer next-token
            # distribution at the last position.
            logits = model.lm_head(model.model.norm(h[:, -1, :]))
            probs = torch.softmax(logits.float(), dim=-1)
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            entropies.append((-(probs * log_probs).sum(-1)).save())
            top2 = probs.topk(2, dim=-1).values
            max_probs.append(top2[:, 0].save())
            margins.append((top2[:, 0] - top2[:, 1]).save())

        final_logits = model.lm_head.output[:, -1, :].save()

    # Everything below runs locally on the saved values.
    print(f"Model: {args.model} ({n_layers} layers)")
    print(f"Prompt: {args.prompt!r}\n")
    print(f"{'layer':>5} {'hidden shape':>20} {'entropy':>9} {'p(top1)':>9} {'margin':>9}")
    for i in range(n_layers):
        print(
            f"{i:>5} {str(tuple(hidden_states[i].shape)):>20} "
            f"{entropies[i].item():>9.3f} {max_probs[i].item():>9.3f} {margins[i].item():>9.3f}"
        )

    final_probs = torch.softmax(final_logits.float(), dim=-1)
    top = final_probs.topk(5, dim=-1)
    print("\nFinal next-token distribution (top 5):")
    for p, idx in zip(top.values[0], top.indices[0]):
        print(f"  {model.tokenizer.decode(idx)!r}: {p.item():.4f}")

    # Example downstream feature vector: layerwise uncertainty profile
    profile = torch.stack([torch.stack(entropies).squeeze(), torch.stack(max_probs).squeeze()])
    print(f"\nUncertainty feature matrix shape (2 x n_layers): {tuple(profile.shape)}")


if __name__ == "__main__":
    main()

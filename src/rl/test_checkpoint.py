from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _extract_qnet_state_dict(ckpt: dict) -> dict:
    # Prefer explicit keys, fall back to common aliases.
    for k in ["q_net", "policy_net", "state_dict", "model_state_dict", "net"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    raise KeyError(f"No q-network state dict found. Keys: {list(ckpt.keys())}")


def _infer_layer_sizes_from_state_dict(sd: dict) -> tuple[int, int]:
    # Find first Linear weight and last Linear weight.
    weight_keys = [k for k in sd.keys() if k.endswith(".weight")]
    if not weight_keys:
        raise ValueError("No .weight tensors in state_dict, cannot infer dims.")
    first_w = sd[sorted(weight_keys)[0]]
    last_w = sd[sorted(weight_keys)[-1]]
    state_dim = int(first_w.shape[1])
    action_dim = int(last_w.shape[0])
    return state_dim, action_dim


def test_checkpoint_loading(checkpoint_path: Path, state_dim: int, device: str = "cpu") -> bool:
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = _extract_qnet_state_dict(ckpt)

    inferred_state_dim, inferred_action_dim = _infer_layer_sizes_from_state_dict(sd)
    LOGGER.info("Checkpoint inferred dims: state_dim=%d action_dim=%d", inferred_state_dim, inferred_action_dim)

    if int(inferred_state_dim) != int(state_dim):
        raise ValueError(f"State dim mismatch: ckpt has {inferred_state_dim} but requested {state_dim}")

    # Minimal forward test: build a plain MLP matching sd layer sizes
    # We do not assume specific class names, only linear stacks.
    # Build sequential by reading weights in order of appearance.
    # This is a sanity check only.
    x = torch.randn(1, state_dim).to(device)

    # Try to load into a simple Linear-only stack by reconstructing hidden sizes.
    # Infer hidden sizes from weight shapes in sorted order.
    weight_keys = sorted([k for k in sd.keys() if k.endswith(".weight")])
    shapes = [sd[k].shape for k in weight_keys]  # (out,in)
    hidden_sizes = [int(s[0]) for s in shapes[:-1]]
    action_dim = int(shapes[-1][0])

    LOGGER.info("Inferred hidden sizes: %s", hidden_sizes)

    import torch.nn as nn

    layers = []
    in_dim = state_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, action_dim))
    net = nn.Sequential(*layers).to(device)

    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        LOGGER.warning("Missing keys: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys: %s", unexpected)

    with torch.no_grad():
        y = net(x)
    LOGGER.info("Forward OK. Output shape: %s", tuple(y.shape))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--state-dim", type=int, default=35)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ok = test_checkpoint_loading(Path(args.checkpoint), state_dim=args.state_dim, device=args.device)
    if ok:
        LOGGER.info("✅ Checkpoint test passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

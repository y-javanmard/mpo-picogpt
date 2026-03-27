"""
benchmark_mpo.py
================
Trains dense PicoGPT and MPO-PicoGPT side-by-side on tiny Shakespeare,
then plots:
  1. Train loss vs steps
  2. Val loss vs steps
  3. Val token accuracy vs steps
  4. Compression summary (params, ratio, final accuracy)

Usage
-----
    # Download tiny Shakespeare first (same source as PicoGPT.jl):
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    # Then run:
    python benchmark_mpo.py [--bond-dims 4 8 16] [--steps 2000] [--eval-every 100]

Requirements: torch, matplotlib  (pip install torch matplotlib)
"""

import argparse
import math
import time
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # headless — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Import the models from the companion file ─────────────────────────────────
# If you renamed mpo_picogpt.py, update this import.
import importlib.util, sys

def _load_mpo_module():
    """Load mpo_picogpt.py from the same directory as this script."""
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "mpo_picogpt", here / "mpo_picogpt.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mpo_picogpt"] = mod
    spec.loader.exec_module(mod)
    return mod

mpo = _load_mpo_module()
GPTConfig      = mpo.GPTConfig
PicoGPT        = mpo.PicoGPT
MPO_PicoGPT    = mpo.MPO_PicoGPT
compress_pretrained = mpo.compress_pretrained
FACTORIZATIONS = mpo.FACTORIZATIONS


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

def get_shakespeare(path: str = "input.txt") -> Tuple[torch.Tensor, dict, dict]:
    """Download (if needed) and tokenise tiny Shakespeare."""
    if not os.path.exists(path):
        print(f"Downloading tiny Shakespeare → {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)

    text = open(path).read()
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for c, i in stoi.items()}
    data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos


def make_splits(data: torch.Tensor, val_frac: float = 0.1):
    n   = len(data)
    cut = int(n * (1 - val_frac))
    return data[:cut], data[cut:]


def get_batch(split: torch.Tensor, seq_len: int,
              batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(split) - seq_len, (batch_size,))
    x  = torch.stack([split[i:i + seq_len]     for i in ix]).to(device)
    y  = torch.stack([split[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module,
             split: torch.Tensor,
             cfg: GPTConfig,
             n_batches: int = 20,
             device: str = "cpu") -> Tuple[float, float]:
    """
    Returns (mean_loss, token_accuracy) on `n_batches` random batches
    drawn from `split`.

    token_accuracy = fraction of next-token predictions that are correct
                     (argmax of logits, position-wise).
    """
    model.eval()
    losses, accs = [], []
    for _ in range(n_batches):
        x, y = get_batch(split, cfg.seq_len, batch_size=8, device=device)
        logits, loss = model(x, y)
        # token accuracy
        preds   = logits.argmax(dim=-1)         # (B, T)
        correct = (preds == y).float().mean()
        losses.append(loss.item())
        accs.append(correct.item())
    model.train()
    return float(np.mean(losses)), float(np.mean(accs))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    """Hyperparameters for one training run."""
    label:       str
    model:       nn.Module
    steps:       int
    batch_size:  int   = 32
    lr:          float = 3e-4
    grad_clip:   float = 1.0
    eval_every:  int   = 100
    warmup:      int   = 100     # linear LR warmup steps


@dataclass
class RunHistory:
    label:       str
    n_params:    int
    steps:       List[int]   = field(default_factory=list)
    train_loss:  List[float] = field(default_factory=list)
    val_loss:    List[float] = field(default_factory=list)
    val_acc:     List[float] = field(default_factory=list)
    wall_times:  List[float] = field(default_factory=list)


def train_run(run_cfg: RunConfig,
              train_data: torch.Tensor,
              val_data:   torch.Tensor,
              gpt_cfg:    GPTConfig,
              device:     str = "cpu") -> RunHistory:
    """Single training run. Returns full history."""
    model = run_cfg.model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr,
                               betas=(0.9, 0.95), weight_decay=0.1)

    def get_lr(step: int) -> float:
        if step < run_cfg.warmup:
            return run_cfg.lr * (step + 1) / run_cfg.warmup
        # cosine decay
        progress = (step - run_cfg.warmup) / max(run_cfg.steps - run_cfg.warmup, 1)
        return run_cfg.lr * (0.5 * (1 + math.cos(math.pi * progress)))

    hist = RunHistory(
        label    = run_cfg.label,
        n_params = sum(p.numel() for p in model.parameters()),
    )

    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"  Training: {run_cfg.label}  ({hist.n_params:,} params)")
    print(f"{'─'*60}")

    for step in range(run_cfg.steps + 1):
        # ── LR schedule ──────────────────────────────────────────────────────
        lr = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # ── Eval ─────────────────────────────────────────────────────────────
        if step % run_cfg.eval_every == 0:
            vl, va = evaluate(model, val_data, gpt_cfg, device=device)
            tl, _  = evaluate(model, train_data, gpt_cfg,
                               n_batches=10, device=device)
            elapsed = time.time() - t0
            hist.steps.append(step)
            hist.train_loss.append(tl)
            hist.val_loss.append(vl)
            hist.val_acc.append(va)
            hist.wall_times.append(elapsed)
            print(f"  step {step:5d}  lr={lr:.2e}  "
                  f"train_loss={tl:.4f}  val_loss={vl:.4f}  "
                  f"val_acc={va:.4f}  ({elapsed:.0f}s)")

        if step == run_cfg.steps:
            break

        # ── Forward + backward ───────────────────────────────────────────────
        x, y = get_batch(train_data, gpt_cfg.seq_len,
                         run_cfg.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), run_cfg.grad_clip)
        opt.step()

    print(f"  Done in {time.time() - t0:.1f}s  "
          f"final val_acc={hist.val_acc[-1]:.4f}")
    return hist


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

# Palette: dense = dark blue, MPO variants in warm tones
COLORS = {
    "Dense":  "#1f4e79",
    "MPO χ=4":  "#e07b39",
    "MPO χ=8":  "#c0392b",
    "MPO χ=16": "#7d3c98",
    "MPO χ=32": "#1a8a5a",
}
LINE_STYLES = {
    "Dense":    "-",
    "MPO χ=4":  "--",
    "MPO χ=8":  "-.",
    "MPO χ=16": ":",
    "MPO χ=32": (0, (3, 1, 1, 1)),
}


def plot_results(histories: List[RunHistory],
                 out_path: str = "benchmark_results.png") -> None:
    """
    Four-panel figure:
      [0,0] Train loss vs step
      [0,1] Val loss vs step
      [1,0] Val token accuracy vs step
      [1,1] Summary bar chart (params + final val accuracy)
    """
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#f8f9fa")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.08, right=0.97, top=0.91, bottom=0.08)

    ax_tl  = fig.add_subplot(gs[0, 0])  # train loss
    ax_vl  = fig.add_subplot(gs[0, 1])  # val loss
    ax_acc = fig.add_subplot(gs[1, 0])  # val accuracy
    ax_bar = fig.add_subplot(gs[1, 1])  # summary bars

    for ax in [ax_tl, ax_vl, ax_acc]:
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#e0e0e0", linewidth=0.6, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Panel 0: Train loss ───────────────────────────────────────────────────
    for h in histories:
        c  = COLORS.get(h.label, "#555")
        ls = LINE_STYLES.get(h.label, "-")
        ax_tl.plot(h.steps, h.train_loss, color=c, lw=2, linestyle=ls,
                   label=h.label)

    ax_tl.set_xlabel("Training step", fontsize=10)
    ax_tl.set_ylabel("Train cross-entropy loss", fontsize=10)
    ax_tl.set_title("Train Loss", fontsize=12, fontweight="bold")
    ax_tl.legend(fontsize=8, framealpha=0.9)

    # ── Panel 1: Val loss ─────────────────────────────────────────────────────
    for h in histories:
        c  = COLORS.get(h.label, "#555")
        ls = LINE_STYLES.get(h.label, "-")
        ax_vl.plot(h.steps, h.val_loss, color=c, lw=2, linestyle=ls,
                   label=h.label)

    ax_vl.set_xlabel("Training step", fontsize=10)
    ax_vl.set_ylabel("Val cross-entropy loss", fontsize=10)
    ax_vl.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax_vl.legend(fontsize=8, framealpha=0.9)

    # ── Panel 2: Val token accuracy ───────────────────────────────────────────
    for h in histories:
        c  = COLORS.get(h.label, "#555")
        ls = LINE_STYLES.get(h.label, "-")
        ax_acc.plot(h.steps, [a * 100 for a in h.val_acc],
                    color=c, lw=2, linestyle=ls, label=h.label)

    ax_acc.set_xlabel("Training step", fontsize=10)
    ax_acc.set_ylabel("Val token accuracy (%)", fontsize=10)
    ax_acc.set_title("Validation Token Accuracy", fontsize=12, fontweight="bold")
    ax_acc.legend(fontsize=8, framealpha=0.9)
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # ── Panel 3: Summary bar chart ────────────────────────────────────────────
    labels     = [h.label for h in histories]
    n_params   = [h.n_params / 1e6 for h in histories]   # millions
    final_accs = [h.val_acc[-1] * 100 for h in histories]
    bar_colors = [COLORS.get(l, "#888") for l in labels]

    x     = np.arange(len(labels))
    width = 0.35

    ax2 = ax_bar.twinx()

    bars1 = ax_bar.bar(x - width / 2, n_params, width,
                       color=bar_colors, alpha=0.85, label="Params (M)")
    bars2 = ax2.bar(x + width / 2, final_accs, width,
                    color=bar_colors, alpha=0.45, hatch="///", label="Val acc (%)")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylabel("Parameters (M)", fontsize=10)
    ax2.set_ylabel("Final val token accuracy (%)", fontsize=10)
    ax_bar.set_title("Parameter Count vs Final Accuracy",
                     fontsize=12, fontweight="bold")
    ax_bar.set_facecolor("#ffffff")
    ax_bar.spines["top"].set_visible(False)

    # Annotate bars with values
    for bar, v in zip(bars1, n_params):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}M", ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(bars2, final_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

    lines1, labs1 = ax_bar.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax_bar.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        "Dense PicoGPT vs MPO-PicoGPT — Shakespeare character prediction",
        fontsize=14, fontweight="bold", y=0.97
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved → {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",       type=int,   default=2000,
                   help="Training steps per run")
    p.add_argument("--eval-every",  type=int,   default=100,
                   help="Evaluate every N steps")
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--bond-dims",   type=int,   nargs="+", default=[4, 8, 16],
                   help="Bond dimensions χ to benchmark")
    p.add_argument("--data",        type=str,   default="input.txt",
                   help="Path to tiny Shakespeare text file")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",         type=str,   default="benchmark_results.png")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--from-pretrained", action="store_true",
                   help="Initialise MPO models by compressing a pretrained dense model")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = GPTConfig(
        vocab_size = 65,    # set correctly after tokenisation below
        d_model    = 128,
        n_heads    = 4,
        n_layers   = 4,
        seq_len    = 256,
        d_ff       = 512,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data, stoi, itos = get_shakespeare(args.data)
    cfg.vocab_size   = len(stoi)
    print(f"Vocabulary size: {cfg.vocab_size}  Dataset: {len(data):,} tokens")
    train_data, val_data = make_splits(data)
    print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")

    # ── Build models ──────────────────────────────────────────────────────────
    print("\nBuilding models...")
    torch.manual_seed(args.seed)
    dense_model = PicoGPT(cfg)

    run_configs = [
        RunConfig(
            label      = "Dense",
            model      = dense_model,
            steps      = args.steps,
            batch_size = args.batch_size,
            lr         = args.lr,
            eval_every = args.eval_every,
        )
    ]

    for chi in args.bond_dims:
        torch.manual_seed(args.seed)

        if args.from_pretrained:
            # Compress the already-trained dense model
            # (call this after dense training — see note below)
            mpo_model = compress_pretrained(dense_model, bond_dim=chi)
        else:
            # Train from scratch with MPO parameterisation
            mpo_model = MPO_PicoGPT(cfg, bond_dim=chi)

        run_configs.append(RunConfig(
            label      = f"MPO χ={chi}",
            model      = mpo_model,
            steps      = args.steps,
            batch_size = args.batch_size,
            lr         = args.lr,
            eval_every = args.eval_every,
        ))

    # ── NOTE on --from-pretrained ─────────────────────────────────────────────
    # If you pass --from-pretrained, dense is trained first, then MPO models
    # are initialised from its weights (TT-SVD compression) and fine-tuned.
    # Without the flag, all models are trained from random initialisations with
    # identical seeds — a fair apples-to-apples comparison.
    #
    # For a compression-then-finetune study, set --steps 500 for MPO runs
    # after compressing a dense model trained for 5000 steps.

    # ── Training ──────────────────────────────────────────────────────────────
    histories = []
    for rc in run_configs:
        torch.manual_seed(args.seed)   # same data shuffle order for all runs
        h = train_run(rc, train_data, val_data, cfg, device=device)
        histories.append(h)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(histories, out_path=args.out)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'Model':<16}  {'Params':>10}  {'Ratio':>7}  {'Val loss':>9}  {'Val acc':>8}")
    print("  " + "-" * 55)
    dense_p = histories[0].n_params
    for h in histories:
        ratio = dense_p / h.n_params
        print(f"  {h.label:<16}  {h.n_params:>10,}  {ratio:>7.1f}×  "
              f"{h.val_loss[-1]:>9.4f}  {h.val_acc[-1]:>8.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()

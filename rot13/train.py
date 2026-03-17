"""
Train a nanochat mini GPT to learn the ROT13 cipher.

Usage
-----
    # From the workspace root:
    python -m dev.rot13.train
    python -m dev.rot13.train --num-steps 5000
    python -m dev.rot13.train --resume           # continue from latest checkpoint
    python -m dev.rot13.train --checkpoint-dir /tmp/rot13
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch

# Allow both `python -m dev.rot13.train` and `python dev/rot13/train.py`
sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.gpt import GPT, GPTConfig


# ── ROT13 helpers ──────────────────────────────────────────────────────────

def rot13_char(c):
    if 'a' <= c <= 'z':
        return chr((ord(c) - ord('a') + 13) % 26 + ord('a'))
    if 'A' <= c <= 'Z':
        return chr((ord(c) - ord('A') + 13) % 26 + ord('A'))
    return c

def rot13(s):  return ''.join(rot13_char(c) for c in s)
def encode(s): return [ord(c) for c in s]

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

def random_word(min_len=3, max_len=8):
    return ''.join(random.choice(ALPHABET) for _ in range(random.randint(min_len, max_len)))

def make_example(word):
    """Build a complete training sequence: 'rot13:hello=uryyb\\n'"""
    return f'rot13:{word}={rot13(word)}\n'


# ── Batch builder ──────────────────────────────────────────────────────────

def build_batch(B: int, T: int, device):
    """
    Returns (inputs, targets) of shape (B, T).

    targets[t] == -1  → prompt position, ignored in cross-entropy loss
    targets[t] == k   → model should predict token k at position t
    """
    inputs  = torch.zeros(B, T, dtype=torch.long)
    targets = torch.full((B, T), -1, dtype=torch.long)
    for i in range(B):
        word  = random_word()
        seq   = make_example(word)
        toks  = encode(seq)
        L     = min(len(toks), T)
        inputs[i, :L] = torch.tensor(toks[:L])
        # Everything up to and including '=' is the prompt;
        # supervision starts at the character after '='.
        plen = len(f'rot13:{word}=')
        for t in range(plen - 1, L - 1):
            targets[i, t] = toks[t + 1]
    return inputs.to(device), targets.to(device)


# ── Training ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train nanochat mini on ROT13')
    # Model architecture
    parser.add_argument('--n-layer',  type=int,   default=2)
    parser.add_argument('--n-embd',   type=int,   default=64)
    parser.add_argument('--n-head',   type=int,   default=2)
    parser.add_argument('--seq-len',  type=int,   default=32)
    # Training
    parser.add_argument('--num-steps',    type=int,   default=3000)
    parser.add_argument('--batch-size',   type=int,   default=128)
    parser.add_argument('--base-lr',      type=float, default=5e-3)
    parser.add_argument('--warmup',       type=int,   default=100, help='LR warmup steps')
    parser.add_argument('--seed',         type=int,   default=0)
    # Checkpointing
    parser.add_argument('--save-every',      type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--checkpoint-dir',  type=str, default='dev/rot13/checkpoints')
    parser.add_argument('--resume',          action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # ── Checkpoint directory ──────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / 'latest.pt'

    # ── Model & optimizer ────────────────────────────────────────────────
    config = GPTConfig(
        vocab_size=128,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_head,
        n_embd=args.n_embd,
        sequence_len=args.seq_len,
        window_pattern='L',
    )
    model     = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.01)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    losses: list[float] = []

    if args.resume and latest.exists():
        print(f'Resuming from {latest}')
        ck         = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_step = ck['step'] + 1
        losses     = ck.get('losses', [])
        print(f'  Resumed at step {start_step}, last loss={losses[-1]:.4f}')
    else:
        model.init_weights()
        n_params = sum(p.numel() for p in model.parameters())
        print(f'Parameters : {n_params:,}')

    # ── LR schedule: warmup → cosine → floor at 3 % of base LR ──────────
    N  = args.num_steps
    W  = args.warmup
    LR = args.base_lr

    def get_lr(step: int) -> float:
        if step < W:
            return 1e-4 + (LR - 1e-4) * step / W
        progress = (step - W) / max(1, N - W)
        return LR * max(0.03, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()

    for step in range(start_step, N):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        inp, tgt = build_batch(args.batch_size, args.seq_len, device)
        loss = model(inp, targets=tgt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == N - 1:
            recent = sum(losses[-50:]) / len(losses[-50:])
            print(f'step {step:4d}/{N}  loss={recent:.4f}  lr={lr:.5f}  elapsed={time.time()-t0:.0f}s')

        # ── Checkpoint ───────────────────────────────────────────────────
        if (step + 1) % args.save_every == 0 or step == N - 1:
            ck = {
                'step':      step,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses':    losses,
                'config':    config,
            }
            numbered = ckpt_dir / f'step_{step:06d}.pt'
            torch.save(ck, numbered)
            torch.save(ck, latest)
            print(f'  [ckpt] saved → {numbered.name}')

    print(f'\nDone in {time.time()-t0:.0f}s  |  checkpoints: {ckpt_dir}')


if __name__ == '__main__':
    main()

"""
Evaluate a trained ROT13 nanochat model.

Usage
-----
    # From the workspace root:
    python -m dev.rot13.eval                         # loads dev/rot13/checkpoints/latest.pt
    python -m dev.rot13.eval --checkpoint path/to/step_002999.pt
    python -m dev.rot13.eval --num-test 1000
    python -m dev.rot13.eval --interactive           # enter your own words
"""

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.gpt import GPT


# ── ROT13 helpers ──────────────────────────────────────────────────────────

def rot13_char(c):
    if 'a' <= c <= 'z':
        return chr((ord(c) - ord('a') + 13) % 26 + ord('a'))
    if 'A' <= c <= 'Z':
        return chr((ord(c) - ord('A') + 13) % 26 + ord('A'))
    return c

def rot13(s):       return ''.join(rot13_char(c) for c in s)
def encode(s):      return [ord(c) for c in s]
def decode(tokens): return ''.join(chr(t) for t in tokens if 0 <= t < 128)

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

def random_word(min_len=3, max_len=8):
    return ''.join(random.choice(ALPHABET) for _ in range(random.randint(min_len, max_len)))


# ── Inference ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict(model, word: str, temperature: float = 0.0) -> str:
    prompt = encode(f'rot13:{word}=')
    out = []
    for tok in model.generate(prompt, max_tokens=len(word) + 2, temperature=temperature):
        if tok == ord('\n'):
            break
        out.append(tok)
        if len(out) > len(word):
            break
    return decode(out)


# ── Accuracy evaluation ────────────────────────────────────────────────────

def evaluate(model, num_test: int, seed: int, temperature: float) -> tuple[int, list]:
    random.seed(seed)
    test_words = [random_word() for _ in range(num_test)]
    correct = 0
    errors = []
    for w in test_words:
        pred = predict(model, w, temperature)
        if pred == rot13(w):
            correct += 1
        else:
            errors.append((w, pred, rot13(w)))
    return correct, errors


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate nanochat mini ROT13 model')
    parser.add_argument('--checkpoint',   type=str,   default='dev/rot13/checkpoints/latest.pt')
    parser.add_argument('--num-test',     type=int,   default=500)
    parser.add_argument('--interactive',  action='store_true', help='Enter words interactively after eval')
    parser.add_argument('--temperature',  type=float, default=0.0, help='0.0 = greedy; >0 = sampling')
    parser.add_argument('--seed',         type=int,   default=999)
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f'Checkpoint not found: {ckpt_path}')
        print('Run  python -m dev.rot13.train  first.')
        return 1

    print(f'Loading {ckpt_path} ...')
    ck     = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ck['config']
    model  = GPT(config).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    n_params      = sum(p.numel() for p in model.parameters())
    trained_steps = ck['step'] + 1
    train_losses  = ck.get('losses', [])
    final_loss    = (sum(train_losses[-50:]) / len(train_losses[-50:])) if train_losses else float('nan')
    print(f'  {n_params:,} params  |  {trained_steps} training steps  |  final loss ≈ {final_loss:.4f}')

    # ── Accuracy ──────────────────────────────────────────────────────────
    print(f'\nEvaluating on {args.num_test} randomly generated words ...')
    correct, errors = evaluate(model, args.num_test, args.seed, args.temperature)
    pct = correct / args.num_test * 100
    print(f'Accuracy: {correct}/{args.num_test} = {pct:.2f}%')

    if errors:
        print(f'\nFirst errors (word → predicted | expected):')
        for w, pred, exp in errors[:10]:
            print(f'  {w:<12} → {pred!r:<14}  (expected {exp!r})')
    else:
        print('No errors — perfect score!')

    # ── Sample predictions ────────────────────────────────────────────────
    print(f'\n{"word":<12} {"predicted":<12} {"expected":<12}')
    print('─' * 42)
    random.seed(42)
    for word in [random_word() for _ in range(20)]:
        pred = predict(model, word, args.temperature)
        ok   = '✓' if pred == rot13(word) else '✗'
        print(f'{word:<12} {pred:<12} {rot13(word):<12} {ok}')

    # ── Interactive mode ──────────────────────────────────────────────────
    if args.interactive:
        print('\nInteractive mode  (Ctrl-C or empty line to exit)')
        while True:
            try:
                word = input('  word> ').strip().lower()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not word:
                break
            pred = predict(model, word, args.temperature)
            exp  = rot13(word)
            ok   = '✓' if pred == exp else '✗'
            print(f'  rot13({word!r}) = {pred!r}   [expected {exp!r}] {ok}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

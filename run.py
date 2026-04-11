"""Single-experiment entrypoint. One (base, p) pair per invocation.

Designed to be called from a SLURM array task that picks a row from manifest.txt.
"""
import argparse
import os
import sys

from config import Config
from train import train_experiment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=int, required=True, help="Numeric base B")
    p.add_argument("--p", type=int, required=True, help="Predict n mod 2^p")
    p.add_argument("--save_dir", type=str, required=True, help="Where to write checkpoints/results")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_resume", action="store_true", help="Ignore existing checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print(f"=== B={args.base} p={args.p} epochs={cfg.n_epochs} save_dir={args.save_dir} ===", flush=True)
    print(f"=== SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '-')} "
          f"SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID', '-')} "
          f"HOST={os.uname().nodename} ===", flush=True)

    train_experiment(
        base=args.base,
        p=args.p,
        save_dir=args.save_dir,
        cfg=cfg,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    sys.exit(main())

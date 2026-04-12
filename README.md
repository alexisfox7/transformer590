# Binary suffix grokking

Predicting `n mod 2^p` from base-B digits with a 4-layer encoder
transformer (Charton & Narayanan architecture). Sweeps 11 bases
× 7 p-values to test whether grokking difficulty tracks ν₂(B),
the 2-adic valuation of the base.

**Authors:** Alexis Fox, Benjamin Greene, Anish Hebbar, Kai Wang

See [GROKKING.md](GROKKING.md) for results and figures.

## Code

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameter dataclass |
| `data.py` | `BinarySuffixDataset`, eval-set builder, `nu_2()` |
| `model.py` | `BinarySuffixTransformer` (encoder-only, mean-pooled) |
| `train.py` | Training loop with checkpointing + early stopping |
| `run.py` | CLI entrypoint: `python run.py --base B --p P --save_dir DIR` |
| `analyze.py` | Generate grokking figures from saved result JSONs |
| `manifest.txt` | Sweep grid (77 configs, one `<base> <p>` per line) |
| `slurm/sweep.sbatch` | SLURM array job for Duke DCC |

## Quick start

```bash
pip install torch numpy matplotlib scikit-learn

# Smoke test (should hit ~100% in 1-2 epochs)
python run.py --base 2 --p 1 --save_dir /tmp/bsfx_test --epochs 2

# Generate figures from saved results
python analyze.py
```

## Cluster sweep

```bash
sbatch slurm/sweep.sbatch   # 77-task array, 2 concurrent GPUs
```

Each task reads its `(base, p)` from `manifest.txt`, trains for up to
400 epochs with early stopping, and writes results to
`$SAVE_DIR/B{B}_p{P}_results.json`. Checkpoints every epoch for clean
resume after preemption (`--requeue`).

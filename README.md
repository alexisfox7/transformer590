# transformer590 — binary suffix grokking sweep

Predicting `n mod 2^p` from base-B digits with a 4-layer encoder transformer,
sweeping `B ∈ {2, 4, 8, 10, 12, 24, 28, 3, 5, 13, 17}` × `p ∈ {1, 2, 3, 4, 5, 6, 8}`
(77 configs).

## Layout

```
config.py        Config dataclass — all hyperparameters
data.py          BinarySuffixDataset, eval-set builder, base helpers
model.py         BinarySuffixTransformer
train.py         train_experiment(): one (B, p) run, checkpointing + early stop
run.py           CLI entrypoint — one (B, p) per invocation
make_manifest.py Generates manifest.txt (77 lines of "<base> <p>")
manifest.txt     Sweep manifest, easier configs first
slurm/sweep.sbatch  SLURM array job for DCC scavenger-h200
binary_suffix_full_experiment.ipynb  Original notebook (kept for reference)
paper/           Overleaf-linked subrepo — DO NOT touch from this repo
```

## Cluster setup (Duke DCC, one-time)

NetID: `af375`. Everything lives on `/work/af375` (NOT `$HOME` — 25GB quota).

```bash
# 1. Install micromamba into /work
cd /work/af375
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=/work/af375/micromamba
eval "$(./bin/micromamba shell hook --shell bash)"

# 2. Create env
micromamba create -p /work/af375/envs/bsfx python=3.11 -c conda-forge -y
micromamba activate /work/af375/envs/bsfx

# 3. Install PyTorch (CUDA 12.x for H200)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib scikit-learn

# 4. Clone the repo into /work
cd /work/af375
git clone git@github.com:alexisfox7/transformer590.git
```

## Running the sweep

```bash
cd /work/af375/transformer590
git pull
python make_manifest.py    # regenerate manifest.txt if you change BASES/P_VALUES
sbatch slurm/sweep.sbatch
```

That submits a 77-task array with at most 2 tasks running concurrently
(one per GPU). Each task:

1. Reads its `(base, p)` from line `$SLURM_ARRAY_TASK_ID` of `manifest.txt`
2. Runs `python run.py --base B --p P --save_dir /work/af375/binary_suffix_experiment --epochs 400`
3. Writes per-epoch checkpoints to `$SAVE_DIR/B{B}_p{P}_checkpoint.pt`
4. Writes final results to `$SAVE_DIR/B{B}_p{P}_results.json`

### Preemption / requeue

Scavenger jobs can be killed by higher-priority work. The sbatch sets
`--requeue`, so SLURM relaunches the same array task on a new node, and
`run.py` resumes from the last per-epoch checkpoint. Already-successful
configs (`best_acc >= 0.90`) are skipped on relaunch.

### Monitoring

```bash
squeue -u af375
tail -f logs/bsfx-<jobid>_<taskid>.out
```

### Pulling results back

```bash
# from your laptop
rsync -av af375@dcc-login.oit.duke.edu:/work/af375/binary_suffix_experiment/ ./results/
```

## Local smoke test

Before submitting to the cluster, sanity-check on CPU with a tiny epoch budget:

```bash
python run.py --base 2 --p 1 --save_dir /tmp/bsfx_test --epochs 2
```

This should converge to ~100% in 1–2 epochs (B=2, p=1 is trivial).

## Changing the sweep

Edit `BASES` / `P_VALUES` in `make_manifest.py`, rerun it, and update the
`--array=1-N%2` line in `slurm/sweep.sbatch` to match the new manifest length.

# Cluster Agent Briefing

You are a Claude Code agent running on a Duke DCC login node. Your job is
to operate this repo on the cluster: set up the environment, submit the
sweep, monitor progress, and debug failures. The user (`af375`, Alexis Fox)
edits code on her laptop and pushes to GitHub. You consume that code on the
cluster — **you do not push back**.

## Project in one paragraph

77-config sweep training a 4-layer encoder transformer to predict
`n mod 2^p` from base-B digits, testing whether grokking difficulty tracks
the 2-adic valuation of the base. Bases `{2, 4, 8, 10, 12, 24, 28, 3, 5, 13, 17}`
× p `{1, 2, 3, 4, 5, 6, 8}`. Uniform 400-epoch budget with early stopping
(stop after `best_acc >= 0.98` plateaus for 4 epochs). Each `(B, p)` is one
SLURM array task.

## Hard rules

1. **Never `git push` from the cluster.** The cluster is a read-only
   consumer. If you find a bug, tell the user — she'll fix it locally and
   push. You `git pull` to sync.
2. **Never put anything in `$HOME`.** It has a 25GB quota that fills fast.
   Code, env, checkpoints, caches all live under `/work/af375/`.
3. **Never `--no-verify`, `--force`, or `git reset --hard`.** Standard
   safety rules apply.
4. **Do not edit `paper/`.** It is an Overleaf-linked subrepo (the user's
   collaborators edit it live on Overleaf). It is `.gitignore`d here, but
   if you ever see it, leave it alone.
5. **Always exclude `dcc-h200-gpu-03`** from job submissions — bad NVLink
   hangs DDP. Already in `slurm/sweep.sbatch`.

## Paths

```
REPO       = /work/af375/transformer590
ENV        = /work/af375/envs/bsfx          (micromamba, python 3.11)
SAVE_DIR   = /work/af375/binary_suffix_experiment
LOGS       = $REPO/logs                     (created by sbatch)
MAMBA_ROOT = /work/af375/micromamba
HF_HOME    = /work/af375/.cache/huggingface
GITHUB     = git@github.com:alexisfox7/transformer590.git
```

## First-time setup checklist

Run these once when the repo doesn't yet exist on the cluster:

```bash
cd /work/af375
test -d transformer590 || git clone git@github.com:alexisfox7/transformer590.git
cd transformer590

# Micromamba (skip if /work/af375/envs/bsfx already exists)
export MAMBA_ROOT_PREFIX=/work/af375/micromamba
if [ ! -d /work/af375/envs/bsfx ]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba create -p /work/af375/envs/bsfx python=3.11 -c conda-forge -y
    micromamba activate /work/af375/envs/bsfx
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install numpy matplotlib scikit-learn
fi

mkdir -p /work/af375/binary_suffix_experiment $REPO/logs
```

**Verify the env works** before submitting:

```bash
srun -p scavenger-h200 --account=scavenger-h200 --gres=gpu:h200:1 \
     --mem=16G --cpus-per-task=4 --time=10:00 \
     --exclude=dcc-h200-gpu-03 --pty bash -c '
cd /work/af375/transformer590
export MAMBA_ROOT_PREFIX=/work/af375/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate /work/af375/envs/bsfx
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python run.py --base 2 --p 1 --save_dir /tmp/bsfx_smoke --epochs 1
'
```

If the smoke test prints an epoch line and `[B2_p1] Done`, the env is good.

## Submitting the sweep

```bash
cd /work/af375/transformer590
git pull
sbatch slurm/sweep.sbatch
```

The sbatch is a 77-task array, `%2` concurrency cap (one task per GPU,
two GPUs available on scavenger). Each task:

1. Reads line `$SLURM_ARRAY_TASK_ID` from `manifest.txt` → `BASE P`
2. Runs `python run.py --base $BASE --p $P --save_dir $SAVE_DIR --epochs 400`
3. Writes `B{BASE}_p{P}_checkpoint.pt` after every epoch
4. Writes `B{BASE}_p{P}_results.json` at completion

If a task is preempted, SLURM `--requeue` puts it back in the queue and
`run.py` resumes from its last checkpoint. Already-successful runs
(`best_acc >= 0.90`) are skipped on relaunch — safe to re-submit the whole
array.

## Monitoring

```bash
squeue -u af375                                           # what's queued/running
sacct -u af375 -X --format=JobID,State,Elapsed,NodeList   # recent jobs + state
tail -f logs/bsfx-<jobid>_<taskid>.out                    # live training output
ls /work/af375/binary_suffix_experiment/*_results.json | wc -l  # how many done
```

To check which configs are still pending:

```bash
cd /work/af375/transformer590
python -c "
import json, os, glob
done = set()
for f in glob.glob('/work/af375/binary_suffix_experiment/B*_results.json'):
    r = json.load(open(f))
    if r.get('completed') and max(r.get('best_acc', 0), r.get('final_acc', 0)) >= 0.90:
        done.add((r['base'], r['p']))
with open('manifest.txt') as f:
    all_configs = [tuple(map(int, l.split())) for l in f if l.strip()]
pending = [c for c in all_configs if c not in done]
print(f'done: {len(done)} / {len(all_configs)}')
print(f'pending: {pending}')
"
```

## Debugging failures

When a task errors out (`sacct` shows `FAILED`):

1. **Read the log first.** `logs/bsfx-<jobid>_<taskid>.out` and `.err`.
2. **Common failures and what to do:**
   - `CUDA out of memory` — large `B` × large `p` blew the budget. Lower
     `--batch_size` in `slurm/sweep.sbatch` (try 512 or 256). Tell the user
     so she can decide whether to commit the change.
   - `ImportError` / missing package — env got out of sync with
     `requirements.txt`. Activate the env and `pip install -r requirements.txt`.
   - `nvjitlink` / `libnvrtc` errors — `LD_LIBRARY_PATH` in the sbatch is
     wrong; check the actual path in `/work/af375/envs/bsfx/lib/python3.11/site-packages/nvidia/`.
   - Job state `PREEMPTED` then auto-`REQUEUED` — expected on scavenger, do
     nothing, it'll resume.
   - Stuck at `dcc-h200-gpu-03` — should never happen (excluded), but if it
     does, `scancel` and resubmit.
3. **Don't fix bugs in the source on the cluster.** If you find one, write
   it up clearly (file:line, what's broken, suggested fix) and stop. The
   user will edit and push from her laptop.
4. **Re-running just the failed configs:** find the task IDs from the
   pending list above, then `sbatch --array=<id1>,<id2>,... slurm/sweep.sbatch`
   (overrides the manifest range from the sbatch directive).

## Output layout

```
/work/af375/binary_suffix_experiment/
├── B{B}_p{P}_checkpoint.pt    per-epoch resume state (heavy)
├── B{B}_p{P}_best_model.pt    best test-acc weights
├── B{B}_p{P}_model.pt         final weights
└── B{B}_p{P}_results.json     metrics + history (this is what you ship back)
```

The user pulls results back to her laptop with:
```bash
rsync -av af375@dcc-login.oit.duke.edu:/work/af375/binary_suffix_experiment/ ./results/
```

You don't need to do that. Just keep the JSONs and `_best_model.pt` files
intact.

## What success looks like

- All 77 array tasks reach state `COMPLETED` (or are skipped because their
  result JSON already shows `best_acc >= 0.90`).
- 77 `B*_results.json` files in `$SAVE_DIR`.
- No `FAILED` jobs in `sacct` for the most recent submission.
- Power-of-2 bases (`2, 4, 8`) hit ~1.0 accuracy in a few epochs.
- Odd bases (`3, 5, 13, 17`) at high `p` are the slowest and may not all
  reach 0.90 — that's a research result, not a failure.

When you're done, report: how many configs completed, which (if any)
didn't reach the success threshold, and total wall-clock time across the
sweep.

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data import BinarySuffixDataset, make_eval_data, max_seq_len, nu_2
from model import BinarySuffixTransformer, count_parameters


def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


@torch.no_grad()
def evaluate_detailed(model, X_eval, y_eval, device, batch_size=2048):
    model.eval()
    num_classes = int(y_eval.max().item()) + 1
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    all_correct = 0
    all_total = 0

    for i in range(0, len(X_eval), batch_size):
        x = X_eval[i : i + batch_size].to(device)
        y = y_eval[i : i + batch_size].to(device)
        preds = model(x).argmax(dim=-1)

        all_correct += (preds == y).sum().item()
        all_total += y.size(0)

        for true_c in range(num_classes):
            mask = y == true_c
            if mask.any():
                pred_c = preds[mask]
                for pc in range(num_classes):
                    confusion[true_c, pc] += (pred_c == pc).sum().item()

    overall_acc = all_correct / all_total
    class_acc = {}
    for c in range(num_classes):
        total = confusion[c].sum().item()
        if total > 0:
            class_acc[c] = confusion[c, c].item() / total
        else:
            class_acc[c] = float("nan")
    return overall_acc, class_acc, confusion


@torch.no_grad()
def evaluate_by_residue_mod_2p(model, X_eval, y_eval, ns_eval, p_check, device, batch_size=2048):
    model.eval()
    mod_check = 2**p_check
    correct_by_residue = defaultdict(int)
    total_by_residue = defaultdict(int)

    for i in range(0, len(X_eval), batch_size):
        x = X_eval[i : i + batch_size].to(device)
        y = y_eval[i : i + batch_size].to(device)
        ns = ns_eval[i : i + batch_size]
        preds = model(x).argmax(dim=-1)
        correct = (preds == y).cpu()
        residues = (ns % mod_check).numpy()

        for j in range(len(ns)):
            r = int(residues[j])
            correct_by_residue[r] += int(correct[j].item())
            total_by_residue[r] += 1

    return {r: correct_by_residue[r] / total_by_residue[r] for r in sorted(total_by_residue) if total_by_residue[r] > 0}


def get_result_score(result):
    if result is None:
        return float("-inf")
    candidates = []
    for key in ("best_acc", "final_acc"):
        val = result.get(key)
        if isinstance(val, (int, float)):
            candidates.append(float(val))
    return max(candidates) if candidates else float("-inf")


def experiment_succeeded(result, threshold):
    return result.get("completed", False) and get_result_score(result) >= threshold


def train_experiment(base: int, p: int, save_dir: str, cfg: Config, resume: bool = True):
    """
    Train a single (base, p) experiment with full logging and checkpointing.
    Designed for single-job invocation on a cluster: writes its own checkpoint
    every epoch and resumes cleanly if relaunched (e.g. after preemption).
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    exp_name = f"B{base}_p{p}"
    ckpt_path = os.path.join(save_dir, f"{exp_name}_checkpoint.pt")
    result_path = os.path.join(save_dir, f"{exp_name}_results.json")
    best_model_path = os.path.join(save_dir, f"{exp_name}_best_model.pt")
    model_path = os.path.join(save_dir, f"{exp_name}_model.pt")

    # Skip if a previous run already succeeded.
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
        if experiment_succeeded(existing, cfg.success_acc_threshold):
            print(f"[{exp_name}] Already successful (score={get_result_score(existing):.4f}). Skipping.")
            return existing

    num_classes = 2**p
    msl = max_seq_len(cfg.n_max, base) + 2
    vocab_size = base + 3
    pad_id = base + 2

    model = BinarySuffixTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        num_layers=cfg.enc_layers,
        dim_feedforward=cfg.dim_ff,
        max_len=msl,
        dropout=cfg.dropout,
        pad_id=pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()

    X_eval, y_eval, ns_eval = make_eval_data(base, p, cfg.n_max, cfg.eval_size, msl, seed=12345)

    history = {
        "epoch": [], "train_loss": [], "train_acc": [], "test_acc": [],
        "class_acc": [], "confusion": [], "examples_seen": [],
    }
    start_epoch = 0
    global_step = 0

    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"[{exp_name}] Resumed from epoch {start_epoch}")

    nparams = count_parameters(model)
    chance = 1.0 / num_classes
    print(f"[{exp_name}] classes={num_classes} | max_len={msl} | params={nparams:,} | chance={chance:.4f} | device={device}")

    exp_seed = cfg.seed + 1000 * base + p
    train_ds = BinarySuffixDataset(base, p, cfg.n_max, msl, seed=exp_seed)

    num_workers = cfg.num_workers if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = max(history["test_acc"]) if history["test_acc"] else -1.0
    epochs_without_improve = 0
    completed_epochs = start_epoch

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = get_lr(global_step, cfg.warmup_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * y.size(0)
            epoch_correct += (logits.argmax(-1) == y).sum().item()
            epoch_total += y.size(0)
            global_step += 1

            if epoch_total >= cfg.epoch_size:
                break

        avg_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        test_acc, class_acc, conf_mat = evaluate_detailed(model, X_eval, y_eval, device)
        elapsed = time.time() - t0
        completed_epochs = epoch + 1

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["class_acc"].append({str(k): v for k, v in class_acc.items()})
        history["confusion"].append(conf_mat.tolist())
        history["examples_seen"].append(
            (history["examples_seen"][-1] if history["examples_seen"] else 0) + epoch_total
        )

        print(
            f"  epoch {epoch+1:>3}/{cfg.n_epochs}  "
            f"loss={avg_loss:.4f}  train={train_acc:.4f}  test={test_acc:.4f}  "
            f"best={best_acc:.4f}  ({elapsed:.0f}s, {epoch_total/max(elapsed,1e-6):.0f} ex/s)",
            flush=True,
        )

        if (epoch + 1) % cfg.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "base": base,
                    "p": p,
                },
                ckpt_path,
            )

        if test_acc > best_acc + cfg.min_delta:
            best_acc = test_acc
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            if best_acc >= cfg.early_stop_acc_threshold:
                epochs_without_improve += 1
            else:
                epochs_without_improve = 0

        if (
            completed_epochs >= cfg.min_epochs
            and best_acc >= cfg.early_stop_acc_threshold
            and epochs_without_improve >= cfg.early_stopping_patience
        ):
            print(
                f"  early stopping at epoch {completed_epochs} "
                f"(best_acc={best_acc:.4f} >= {cfg.early_stop_acc_threshold:.2f})"
            )
            break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_acc, final_class_acc, _ = evaluate_detailed(model, X_eval, y_eval, device)

    residue_analysis = {}
    if p >= 2:
        for p_check in range(1, min(p + 2, 9)):
            acc_by_res = evaluate_by_residue_mod_2p(model, X_eval, y_eval, ns_eval, p_check, device)
            residue_analysis[f"mod_2^{p_check}"] = acc_by_res

    torch.save(model.state_dict(), model_path)

    success = best_acc >= cfg.success_acc_threshold
    results = {
        "base": base, "p": p, "num_classes": num_classes,
        "final_acc": final_acc,
        "final_class_acc": {str(k): v for k, v in final_class_acc.items()},
        "best_acc": best_acc,
        "success": success,
        "success_threshold": cfg.success_acc_threshold,
        "early_stop_threshold": cfg.early_stop_acc_threshold,
        "epochs_completed": completed_epochs,
        "history": history,
        "residue_analysis": {
            k: {str(kk): vv for kk, vv in v.items()} for k, v in residue_analysis.items()
        },
        "params": nparams,
        "completed": True,
        "nu_2_B": nu_2(base),
    }

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    status = "SUCCESS" if success else "NEEDS RETRY"
    print(f"[{exp_name}] Done. final={final_acc:.4f} best={best_acc:.4f} [{status}]")
    return results

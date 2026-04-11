from dataclasses import dataclass


@dataclass
class Config:
    # Data generation
    n_max: int = 10**8
    epoch_size: int = 300_000
    eval_size: int = 50_000

    # Architecture (matching Charton & Narayanan)
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 4
    dim_ff: int = 2048
    dropout: float = 0.1

    # Training
    n_epochs: int = 400
    batch_size: int = 1024
    lr: float = 3e-4
    warmup_steps: int = 500
    grad_clip: float = 5.0

    # Early stopping
    success_acc_threshold: float = 0.90
    early_stop_acc_threshold: float = 0.98
    min_epochs: int = 5
    early_stopping_patience: int = 4
    min_delta: float = 1e-3

    # Logging / checkpointing
    save_every: int = 1
    eval_every: int = 1
    log_interval: int = 200

    # Misc
    seed: int = 1337
    num_workers: int = 2

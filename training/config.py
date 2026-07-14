from dataclasses import dataclass

@dataclass
class TrainerConfig:
    # Training
    epochs: int = 250
    batch_size: int = 2

    # Optimization
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    # Precision
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"

    # Performance
    compile_model: bool = True
    channels_last: bool = True

    # Logging
    log_freq: int = 10
    save_dir: str = "checkpoints"

    # Early stopping
    patience: int = 30

    # Optional features
    enable_profiler: bool = False
    ema_enabled: bool = False
    swa_enabled: bool = False
    disable_dashboard: bool = False

    # Full Config Storage
    full_config: dict = None

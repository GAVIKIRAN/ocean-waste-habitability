"""Train the YOLO11 underwater-waste detector.

Features:
  * GPU auto-detection (``device = 0 if torch.cuda.is_available() else "cpu"``), printed at start.
  * ``--mode demo`` (yolo11n, few epochs, minutes) or ``--mode full`` (config defaults: yolo11s, 100 epochs).
  * ``--low-vram``: yolo11n, imgsz 512, batch 4.
  * CUDA out-of-memory -> batch is halved and training retried automatically; when batch
    reaches 1 the fallback model (yolo11n) is used.
  * ``--tune``: hyperparameter search (Optuna TPE or Ultralytics' genetic tuner) before the final run.
  * Adam optimiser, early stopping (``patience``), periodic checkpoints (``save_period``), AMP.

Usage::

    python -m src.waste.train --mode demo
    python -m src.waste.train --mode full [--low-vram] [--batch -1] [--tune]
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

from src.data.prepare_yolo_dataset import write_dataset_yaml
from src.utils.common import describe_device, ensure_dir, get_device, get_logger, load_config, resolve_path, safe_workers, set_seed

log = get_logger(__name__)

AUG_KEYS = ("fliplr", "flipud", "degrees", "translate", "scale", "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup", "close_mosaic")


def is_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory error (any of torch's flavours)."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "cublas_status_alloc_failed" in msg


def next_batch_after_oom(batch: int) -> int:
    """Batch size to retry with after an OOM (auto-batch ``-1`` falls back to 8)."""
    return 8 if batch <= 0 else batch // 2


def resolve_weights(name: str) -> str:
    """Prefer ``models/<name>``; let Ultralytics download pretrained weights there.

    Falls back to the architecture YAML (random init) if the download is impossible.
    """
    local = resolve_path("models") / Path(name).name
    if local.exists() or Path(name).is_absolute():
        return str(local if local.exists() else name)
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        return str(attempt_download_asset(local))
    except Exception as exc:  # noqa: BLE001
        yaml_name = Path(name).with_suffix(".yaml").name
        log.warning("Could not fetch pretrained %s (%s) - training from scratch with %s", name, exc, yaml_name)
        return yaml_name


def build_train_args(cfg: dict[str, Any], mode: str, low_vram: bool, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge config train section + mode preset + low-VRAM preset + CLI overrides."""
    t = dict(cfg["waste"]["train"])
    if mode == "demo":
        t.update(cfg["waste"]["demo"])
    if low_vram:
        t.update(cfg["waste"]["low_vram"])
    t.update({k: v for k, v in overrides.items() if v is not None})
    return t


def _ultra_kwargs(t: dict[str, Any], data_yaml: Path, device: int | str, run_name: str | None = None, seed: int = 42) -> dict[str, Any]:
    kw = dict(
        data=str(data_yaml),
        epochs=int(t["epochs"]),
        patience=int(t["patience"]),
        imgsz=int(t["imgsz"]),
        batch=int(t["batch"]),
        workers=safe_workers(int(t["workers"])),
        optimizer=t["optimizer"],
        lr0=float(t["lr0"]),
        lrf=float(t["lrf"]),
        weight_decay=float(t["weight_decay"]),
        amp=bool(t["amp"]) and device != "cpu",
        cache=t["cache"],
        save_period=int(t.get("save_period", -1)),
        device=device,
        project=str(resolve_path(t["project"])),
        name=run_name or t["name"],
        exist_ok=True,
        seed=seed,
        deterministic=True,
        plots=True,
        verbose=False,
    )
    kw.update({k: t[k] for k in AUG_KEYS if k in t})
    return kw


def train_with_oom_retry(model_name: str, fallback_model: str, kwargs: dict[str, Any], max_retries: int = 6) -> tuple[Any, dict[str, Any]]:
    """Run ``YOLO(model).train(**kwargs)``; on CUDA OOM halve the batch and retry.

    Returns ``(trained YOLO model, the kwargs that finally succeeded)``.
    """
    from ultralytics import YOLO
    import torch

    current_model, kw = model_name, dict(kwargs)
    for attempt in range(max_retries + 1):
        model = YOLO(resolve_weights(current_model))
        try:
            log.info("Training %s | imgsz=%s batch=%s epochs=%s (attempt %d)", current_model, kw["imgsz"], kw["batch"], kw["epochs"], attempt + 1)
            model.train(**kw)
            return model, {**kw, "model": current_model}
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc) or attempt == max_retries:
                raise
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_batch = next_batch_after_oom(int(kw["batch"]))
            if new_batch < 1:
                if current_model == fallback_model:
                    raise RuntimeError("CUDA OOM even at batch=1 with the fallback model; try --low-vram or a smaller imgsz") from exc
                log.warning("CUDA OOM at batch 1 - switching to fallback model %s", fallback_model)
                current_model, new_batch = fallback_model, 4
            log.warning("CUDA out of memory - retrying with batch=%d", new_batch)
            kw["batch"] = new_batch
    raise RuntimeError("unreachable")


def tune_optuna(model_name: str, base_kw: dict[str, Any], n_trials: int, epochs: int, out_dir: Path) -> dict[str, Any]:
    """Bayesian (TPE) search over learning rate and augmentation params; maximises mAP50-95."""
    import optuna
    from ultralytics import YOLO

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
            "scale": trial.suggest_float("scale", 0.2, 0.9),
            "degrees": trial.suggest_float("degrees", 0.0, 20.0),
            "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.05),
            "hsv_s": trial.suggest_float("hsv_s", 0.3, 0.9),
            "hsv_v": trial.suggest_float("hsv_v", 0.2, 0.6),
            "fliplr": trial.suggest_float("fliplr", 0.2, 0.7),
            "mosaic": trial.suggest_float("mosaic", 0.5, 1.0),
            "mixup": trial.suggest_float("mixup", 0.0, 0.3),
        }
        kw = {**base_kw, **params, "epochs": epochs, "patience": epochs, "name": f"tune_{trial.number}", "plots": False, "save_period": -1}
        model = YOLO(resolve_weights(model_name))
        try:
            metrics = model.train(**kw)
        except Exception as exc:  # noqa: BLE001
            if is_oom(exc):
                raise optuna.TrialPruned("CUDA OOM") from exc
            raise
        return float(metrics.box.map) if metrics is not None else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    result = {"backend": "optuna", "best_value_map50_95": study.best_value, "best_params": study.best_params, "n_trials": len(study.trials)}
    (out_dir / "tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Optuna best mAP50-95=%.4f params=%s", study.best_value, study.best_params)
    return study.best_params


def tune_ultralytics(model_name: str, base_kw: dict[str, Any], n_trials: int, epochs: int, out_dir: Path) -> dict[str, Any]:
    """Ultralytics' built-in genetic tuner (``model.tune``)."""
    import yaml
    from ultralytics import YOLO

    model = YOLO(resolve_weights(model_name))
    kw = {k: v for k, v in base_kw.items() if k not in ("name", "exist_ok", "save_period")}
    kw.update(epochs=epochs, iterations=n_trials, plots=False, save=False, val=True, name="tune_ultralytics")
    model.tune(**kw)
    best_yaml = sorted(resolve_path(base_kw["project"]).glob("tune_ultralytics*/best_hyperparameters.yaml"))
    best = yaml.safe_load(best_yaml[-1].read_text()) if best_yaml else {}
    (out_dir / "tuning.json").write_text(json.dumps({"backend": "ultralytics", "best_params": best}, indent=2), encoding="utf-8")
    return best or {}


def export_artifacts(model: Any, cfg: dict[str, Any], summary: dict[str, Any]) -> Path:
    """Copy best.pt to models/, and training curves/CSV to reports/waste/training/."""
    save_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best) if Path(model.trainer.best).exists() else save_dir / "weights" / "last.pt"
    dst = resolve_path(cfg["waste"]["weights"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    rep = ensure_dir(Path(cfg["paths"]["reports"]) / "waste" / "training")
    for f in list(save_dir.glob("*.png")) + list(save_dir.glob("*.jpg")) + list(save_dir.glob("results.csv")) + list(save_dir.glob("args.yaml")):
        shutil.copy2(f, rep / f.name)
    (rep / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("Best weights -> %s ; training artefacts -> %s", dst, rep)
    return dst


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["demo", "full"], default="demo")
    ap.add_argument("--low-vram", action="store_true", help="yolo11n, imgsz 512, batch 4")
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="-1 = auto-batch")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=None, help="false | disk | ram")
    ap.add_argument("--device", default=None, help="override: 0 | cpu")
    ap.add_argument("--tune", action="store_true", help="hyperparameter search before final training")
    ap.add_argument("--tune-backend", choices=["optuna", "ultralytics"], default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device: int | str = get_device() if args.device is None else (int(args.device) if str(args.device).isdigit() else args.device)
    print(f"[waste/train] Device: {describe_device(device)}", flush=True)

    cache = None if args.cache is None else (False if args.cache.lower() == "false" else args.cache)
    t = build_train_args(cfg, args.mode, args.low_vram, {"model": args.model, "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz, "workers": args.workers, "cache": cache})
    data_yaml = resolve_path(cfg["waste"]["dataset_yaml"])
    if not (resolve_path(cfg["waste"]["dataset_dir"]) / "images" / "train").exists():
        raise SystemExit("Processed dataset missing. Run: python -m src.data.download_data && python -m src.data.prepare_yolo_dataset")
    write_dataset_yaml(cfg)  # refresh absolute path (portable across machines)

    rep_dir = ensure_dir(Path(cfg["paths"]["reports"]) / "waste")
    kw = _ultra_kwargs(t, data_yaml, device, seed=cfg["project"]["seed"])
    if args.tune:
        tc = cfg["waste"]["tune"]
        n_trials = tc["demo_n_trials"] if args.mode == "demo" else tc["n_trials"]
        ep = tc["demo_epochs_per_trial"] if args.mode == "demo" else tc["epochs_per_trial"]
        backend = args.tune_backend or tc["backend"]
        tuner = tune_optuna if backend == "optuna" else tune_ultralytics
        best = tuner(t["model"], kw, n_trials, ep, rep_dir)
        kw.update({k: v for k, v in best.items() if k in kw or k in AUG_KEYS or k == "lr0"})

    start = time.time()
    model, used = train_with_oom_retry(t["model"], t["fallback_model"], kw)
    summary = {
        "mode": args.mode,
        "device": describe_device(device),
        "model": used["model"],
        "imgsz": used["imgsz"],
        "batch": used["batch"],
        "epochs_requested": used["epochs"],
        "duration_min": round((time.time() - start) / 60, 2),
        "save_dir": str(model.trainer.save_dir),
    }
    try:
        m = model.trainer.metrics or {}
        summary["final_val_metrics"] = {k: float(v) for k, v in m.items()}
    except Exception:  # noqa: BLE001
        pass
    export_artifacts(model, cfg, summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

"""
Training Loop — GPU Optimized for H200 / A100
- torch.compile() for graph optimization
- AMP (mixed precision float16) — 2-3x speedup on H200
- Large batch size (4096) — saturates GPU memory bandwidth
- Per-batch progress + ETA
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, time, sys
from pathlib import Path
from model import BTCANN

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU = torch.cuda.get_device_name(0)
    print(f"[train] device=cuda  gpu={GPU}")
    print(f"[train] VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
else:
    DEVICE = torch.device("cpu")
    print("[train] device=cpu  (no GPU found)")

USE_AMP = DEVICE.type == "cuda"

CFG = {
    "batch_size": 4096,  # H200 can handle much larger — fills GPU
    "epochs": 150,
    "lr": 3e-3,  # higher LR works well with large batches
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "patience": 15,
    "n_classes": 3,
}

# ─── Data ─────────────────────────────────────────────────────────────────────


def load_data():
    X_train = np.array(np.load("data/processed/X_train.npy"), dtype=np.float32)
    y_train = np.array(np.load("data/processed/y_train.npy"), dtype=np.int64)
    X_val = np.array(np.load("data/processed/X_val.npy"), dtype=np.float32)
    y_val = np.array(np.load("data/processed/y_val.npy"), dtype=np.int64)
    X_test = np.array(np.load("data/processed/X_test.npy"), dtype=np.float32)
    y_test = np.array(np.load("data/processed/y_test.npy"), dtype=np.int64)
    with open("data/processed/feature_cols.json") as f:
        feature_cols = json.load(f)
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = 1.0 / (counts + 1e-9)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ─── Progress bar ─────────────────────────────────────────────────────────────


def progress(i, total, loss, width=35):
    pct = i / total
    done = int(width * pct)
    bar = "█" * done + "░" * (width - done)
    sys.stdout.write(f"\r  [{bar}] {i}/{total}  loss={loss:.4f}")
    sys.stdout.flush()


# ─── Train ────────────────────────────────────────────────────────────────────


def train():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    n_features = X_train.shape[1]
    print(
        f"[train] n_features={n_features}  train={len(X_train):,}  "
        f"val={len(X_val):,}  test={len(X_test):,}"
    )

    # Pin memory only on CUDA
    pin = DEVICE.type == "cuda"

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=4 if pin else 0,
        pin_memory=pin,
        persistent_workers=pin,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"] * 2,
        shuffle=False,
        num_workers=4 if pin else 0,
        pin_memory=pin,
        persistent_workers=pin,
    )

    model = BTCANN(
        n_features=n_features, n_classes=CFG["n_classes"], dropout=CFG["dropout"]
    ).to(DEVICE)

    # torch.compile — huge speedup on H200 (PyTorch 2.x)
    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        print("[train] compiling model with torch.compile...")
        model = torch.compile(model)

    print(f"[train] params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y_train, CFG["n_classes"])
    )
    optimizer = AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CFG["lr"],
        steps_per_epoch=len(train_dl),
        epochs=CFG["epochs"],
        pct_start=0.1,
    )

    # AMP scaler for float16 on CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    Path("models").mkdir(exist_ok=True)
    best_val_f1 = 0.0
    patience_cnt = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    print(
        f"[train] batches/epoch={len(train_dl)}  "
        f"batch={CFG['batch_size']}  AMP={USE_AMP}\n"
    )

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for i, (xb, yb) in enumerate(train_dl, 1):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = criterion(model(xb), yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * len(xb)
            if i % 5 == 0 or i == len(train_dl):
                progress(i, len(train_dl), loss.item())

        sys.stdout.write("\r" + " " * 70 + "\r")
        train_loss /= len(train_ds)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(
                    DEVICE, non_blocking=True
                )
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        val_loss /= len(val_ds)

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        star = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            # Save unwrapped model if compiled
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": raw.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "n_features": n_features,
                    "cfg": CFG,
                },
                "models/best_model.pt",
            )
            star = "  ✓ best"
        else:
            patience_cnt += 1

        eta = (CFG["epochs"] - epoch) * elapsed / 60
        print(
            f"Epoch {epoch:3d}/{CFG['epochs']} | "
            f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"f1={val_f1:.4f}  acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)  ETA={eta:.1f}m{star}"
        )

        if patience_cnt >= CFG["patience"]:
            print(f"\n[early stop] no improvement for {CFG['patience']} epochs")
            break

    # ── Test eval ─────────────────────────────────────────────────────────────
    print("\n=== Test Set Evaluation ===")
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = torch.load("models/best_model.pt", map_location=DEVICE, weights_only=False)
    raw.load_state_dict(ckpt["model_state"])
    raw.eval()

    with torch.no_grad():
        test_probs = (
            raw.predict_proba(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
            .cpu()
            .numpy()
        )
    test_preds = test_probs.argmax(1)

    print(
        classification_report(
            y_test, test_preds, target_names=["SELL", "HOLD", "BUY"], digits=4
        )
    )

    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greys",
        xticklabels=["SELL", "HOLD", "BUY"],
        yticklabels=["SELL", "HOLD", "BUY"],
    )
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("data/processed/confusion_matrix.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["val_f1"])
    axes[1].set_title("Val F1")
    axes[2].plot(history["val_acc"])
    axes[2].set_title("Val Accuracy")
    plt.tight_layout()
    plt.savefig("data/processed/training_curves.png", dpi=120)
    plt.close()

    print(f"\n[done] best val_f1={best_val_f1:.4f}")
    print("Saved: models/best_model.pt")


if __name__ == "__main__":
    train()

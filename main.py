import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from src.dataset import ZipTrainDataset
from src.inference import predict_from_zip
from src.models import XceptionResNet50
from src.trainer import run_epoch, calculate_metrics, count_parameters
from src.transforms import get_train_transforms, get_val_transforms


def prepare_dataloaders():
    print(f"IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}")

    full_train_dataset = ZipTrainDataset(DATASET_ZIP, transform=get_train_transforms(IMG_SIZE))

    total_samples = len(full_train_dataset)
    indices = list(range(len(full_train_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    test_indices = indices[:total_samples]

    # Class weights
    selected_labels = full_train_dataset.labels[test_indices].numpy()
    class_counts = np.bincount(selected_labels)
    total = len(selected_labels)
    class_weights = torch.tensor([total / (2 * count) for count in class_counts],
                                 dtype=torch.float32).to(DEVICE)
    print(f"Using {total_samples} samples for fast test (instead of {len(full_train_dataset)})")
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    # Train/Val split
    train_size = int(0.8 * total_samples)
    train_indices = test_indices[:train_size]
    val_indices = test_indices[train_size:]

    train_dataset_full = ZipTrainDataset(DATASET_ZIP, transform=get_train_transforms(IMG_SIZE))
    val_dataset_full = ZipTrainDataset(DATASET_ZIP, transform=get_val_transforms(IMG_SIZE))

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    # Disable pin_memory for MPS (not supported)
    pin_memory = DEVICE.type != 'mps'
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader, class_weights


def build_model(class_weights):
    model = XceptionResNet50(num_classes=2).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # RESUME TRAINING
    start_epoch = 0
    resume_path = RESUME_PATH

    if os.path.exists(resume_path):
        print(f"🔁 Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)

        model = XceptionResNet50(num_classes=2).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        best_f1 = checkpoint.get("val_f1", 0.0)
        start_epoch = checkpoint.get("epoch", 0)

        model.to(DEVICE)
        print(f"✅ Loaded checkpoint from epoch {start_epoch}, best_f1={best_f1:.4f}")
    else:
        print("🚀 No checkpoint found, starting from scratch.")
        start_epoch = 0
        best_f1 = 0.0

    return model, loss_fn, optimizer, scheduler, start_epoch, best_f1


def train_model(model, loss_fn, optimizer, scheduler, start_epoch, best_f1, train_loader, val_loader):
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    best_model_path = BEST_MODEL_PATH

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_loss, train_f1, train_prec, train_rec = run_epoch(
            DEVICE, model, optimizer, loss_fn, train_loader,
            is_train=True, return_predictions=True
        )
        train_losses.append(train_loss)
        train_f1_scores.append(train_f1)

        # Validation
        val_loss, val_f1, val_prec, val_rec = run_epoch(
            DEVICE, model, optimizer, loss_fn, val_loader,
            is_train=False, return_predictions=True
        )
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        # Update learning rate
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"\nTrain - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, "
              f"Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
        print(f"⏱️  Epoch time: {epoch_time/60:.1f} min, Total: {total_time/60:.1f} min")

        # Save checkpoint
        epoch_model_path = f"epoch_{epoch+1:03d}_xception.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "val_loss": val_loss,
        }, epoch_model_path)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_loss": val_loss,
            }, best_model_path)
            print(f"✓ Saved best model with F1: {best_f1:.4f}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training completed in {total_time/60:.1f} minutes!")
    print(f"🏆 Best F1: {best_f1:.4f}")
    print(f"{'='*60}")


def run_inference():
    print("\n" + "="*60)
    predict_from_zip(DATASET_ZIP, BEST_MODEL_PATH, IMG_SIZE, str(DEVICE), OUTPUT_CSV)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["with_training", "only_inference"],
        default="with_training",
        help="run mode: full training with inference or only inference",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.mode == "with_training":
        train_loader, val_loader, class_weights = prepare_dataloaders()
        model, loss_fn, optimizer, scheduler, start_epoch, best_f1 = build_model(class_weights)
        train_model(model, loss_fn, optimizer, scheduler, start_epoch, best_f1, train_loader, val_loader)
    elif args.mode == "only_inference":
        run_inference()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
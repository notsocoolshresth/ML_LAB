"""
Training pipeline for Landslide Segmentation.

Usage:
    python train.py --data_dir /path/to/Landslide4sense --epochs 100 --batch_size 16

Downloads dataset automatically from HuggingFace if not present.
"""

import argparse
import copy
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model
from losses import CombinedLoss, BoundaryAwareLoss


def download_dataset(target_dir):
    """Download dataset from HuggingFace."""
    from huggingface_hub import snapshot_download
    if os.path.exists(os.path.join(target_dir, "images")):
        print("Dataset already downloaded.")
        return target_dir
    print("Downloading Landslide4Sense dataset...")
    path = snapshot_download(
        repo_id="ibm-nasa-geospatial/Landslide4sense",
        repo_type="dataset",
        local_dir=target_dir,
    )
    print(f"Dataset downloaded to: {path}")
    return path


def compute_metrics(preds, targets, threshold=0.5):
    """Compute P, R, F1, IoU from binary predictions."""
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    targets_bin = (targets > 0.5).float()

    tp = (preds_bin * targets_bin).sum().item()
    fp = (preds_bin * (1 - targets_bin)).sum().item()
    fn = ((1 - preds_bin) * targets_bin).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def _extract_main_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _apply_spectral_perturbation(images, drop_prob=0.0, max_drop=0, noise_std=0.0):
    if drop_prob <= 0.0 and noise_std <= 0.0:
        return images

    augmented = images.clone()
    b, c, _, _ = augmented.shape

    if drop_prob > 0.0 and max_drop > 0:
        for i in range(b):
            if torch.rand(1, device=augmented.device).item() < drop_prob:
                n_drop = int(torch.randint(1, max_drop + 1, (1,), device=augmented.device).item())
                drop_idx = torch.randperm(c, device=augmented.device)[:n_drop]
                augmented[i, drop_idx] = 0.0

    if noise_std > 0.0:
        augmented = augmented + noise_std * torch.randn_like(augmented)

    return augmented


def _update_ema_model(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.data.copy_(buffer.data)


def _forward_with_tta(model, images, use_tta=False):
    if not use_tta:
        return _extract_main_logits(model(images))

    transforms = [
        (lambda x: x, lambda y: y),
        (lambda x: torch.flip(x, dims=[3]), lambda y: torch.flip(y, dims=[3])),
        (lambda x: torch.flip(x, dims=[2]), lambda y: torch.flip(y, dims=[2])),
        (lambda x: torch.flip(x, dims=[2, 3]), lambda y: torch.flip(y, dims=[2, 3])),
    ]

    logits_sum = None
    for aug_fn, inv_fn in transforms:
        aug_images = aug_fn(images)
        aug_logits = _extract_main_logits(model(aug_images))
        logits = inv_fn(aug_logits)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    return logits_sum / float(len(transforms))


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    ema_model=None,
    ema_decay=0.995,
    consistency_loader=None,
    consistency_weight=0.0,
    consistency_mode="pseudo",
    consistency_warmup_epochs=10,
    consistency_conf_low=0.2,
    consistency_conf_high=0.8,
    perturb_prob=0.0,
    perturb_max_drop=0,
    perturb_noise_std=0.0,
):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    consistency_iter = iter(consistency_loader) if consistency_loader is not None else None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        sup_loss = criterion(outputs, masks)
        loss = sup_loss
        cons_loss_value = 0.0

        ramp = 0.0
        if epoch >= consistency_warmup_epochs:
            ramp = min(1.0, (epoch - consistency_warmup_epochs + 1) / 5.0)
        effective_consistency_weight = consistency_weight * ramp

        if consistency_iter is not None and effective_consistency_weight > 0.0:
            try:
                u_images, _ = next(consistency_iter)
            except StopIteration:
                consistency_iter = iter(consistency_loader)
                u_images, _ = next(consistency_iter)

            u_images = u_images.to(device)
            student_u = _apply_spectral_perturbation(
                u_images,
                drop_prob=perturb_prob,
                max_drop=perturb_max_drop,
                noise_std=perturb_noise_std,
            )

            student_logits = _extract_main_logits(model(student_u))
            teacher_net = ema_model if ema_model is not None else model
            with torch.no_grad():
                teacher_logits = _extract_main_logits(teacher_net(u_images))

            teacher_probs = torch.sigmoid(teacher_logits)
            if consistency_mode == "mse":
                cons_loss = F.mse_loss(torch.sigmoid(student_logits), teacher_probs)
            else:
                pseudo_targets = (teacher_probs > 0.5).float()
                conf_mask = (teacher_probs < consistency_conf_low) | (teacher_probs > consistency_conf_high)
                bce_map = F.binary_cross_entropy_with_logits(
                    student_logits, pseudo_targets, reduction="none"
                )
                if conf_mask.any():
                    cons_loss = (bce_map * conf_mask.float()).sum() / conf_mask.float().sum().clamp_min(1.0)
                else:
                    cons_loss = torch.zeros((), device=student_logits.device)

            cons_loss_value = cons_loss.item()
            loss = loss + effective_consistency_weight * cons_loss

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if ema_model is not None:
            _update_ema_model(ema_model, model, ema_decay)

        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        running_loss += loss.item()
        main_logits = _extract_main_logits(outputs)
        all_preds.append(main_logits.detach().cpu())
        all_targets.append(masks.detach().cpu())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            sup=f"{sup_loss.item():.4f}",
            cons=f"{cons_loss_value:.4f}",
            cw=f"{effective_consistency_weight:.3f}",
        )

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    avg_loss = running_loss / len(loader)

    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, masks in tqdm(loader, desc="[Val]"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        logits = _forward_with_tta(model, images, use_tta=use_tta) if use_tta else _extract_main_logits(outputs)

        running_loss += loss.item()
        all_preds.append(logits.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    avg_loss = running_loss / len(loader)

    return avg_loss, metrics


@torch.no_grad()
def evaluate_test(model, loader, device, threshold=0.5, use_tta=False):
    """Evaluate on test set with metrics."""
    model.eval()
    all_preds, all_targets = [], []

    for images, masks in tqdm(loader, desc="[Test]"):
        images = images.to(device)
        outputs = _forward_with_tta(model, images, use_tta=use_tta)
        all_preds.append(outputs.cpu())
        all_targets.append(masks)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Check if test labels are all zeros (no labels provided)
    if all_targets.sum() == 0:
        print("WARNING: Test labels are all zeros. Metrics may not be meaningful.")
        print("Use the official evaluation server for test metrics.")

    metrics = compute_metrics(all_preds, all_targets, threshold)
    return metrics


def find_best_threshold(model, loader, device, use_tta=False):
    """Search for optimal threshold on validation set."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = _forward_with_tta(model, images, use_tta=use_tta)
            all_preds.append(outputs.cpu())
            all_targets.append(masks)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.3, 0.7, 0.02):
        m = compute_metrics(all_preds, all_targets, threshold=t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = t

    print(f"Best threshold: {best_thresh:.2f} → F1={best_f1:.4f}")
    return best_thresh


def main():
    parser = argparse.ArgumentParser(description="Train Landslide Segmentation Model")
    parser.add_argument("--data_dir", type=str, default="./data/Landslide4sense",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--encoder", type=str, default="resnet34",
                        choices=["resnet34", "resnet50", "efficientnet-b3"])
    parser.add_argument("--model_type", type=str, default="unetpp",
                        choices=["unetpp", "custom"])
    parser.add_argument("--no_indices", action="store_true",
                        help="Disable handcrafted spectral indices (use 14ch only)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from HuggingFace")
    parser.add_argument("--boundary_aux", action="store_true",
                        help="Enable boundary auxiliary head and loss")
    parser.add_argument("--boundary_weight", type=float, default=0.1,
                        help="Weight for boundary auxiliary loss")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA model for evaluation and teacher consistency")
    parser.add_argument("--ema_decay", type=float, default=0.995,
                        help="EMA decay")
    parser.add_argument("--consistency_weight", type=float, default=0.0,
                        help="Weight for unsupervised consistency loss")
    parser.add_argument("--consistency_mode", type=str, default="pseudo",
                        choices=["pseudo", "mse"],
                        help="Consistency objective: confidence-masked pseudo labels or MSE")
    parser.add_argument("--consistency_warmup_epochs", type=int, default=10,
                        help="Epoch to start consistency regularization")
    parser.add_argument("--consistency_conf_low", type=float, default=0.2,
                        help="Lower confidence threshold for pseudo-label consistency mask")
    parser.add_argument("--consistency_conf_high", type=float, default=0.8,
                        help="Upper confidence threshold for pseudo-label consistency mask")
    parser.add_argument("--consistency_split", type=str, default="none",
                        choices=["none", "validation", "test"],
                        help="Unlabeled split for consistency regularization")
    parser.add_argument("--consistency_drop_prob", type=float, default=0.1,
                        help="Probability to apply channel-drop perturbation for consistency")
    parser.add_argument("--consistency_max_drop", type=int, default=1,
                        help="Maximum number of dropped channels for consistency perturbation")
    parser.add_argument("--consistency_noise_std", type=float, default=0.005,
                        help="Gaussian noise std for consistency perturbation")
    parser.add_argument("--tta", action="store_true",
                        help="Use flip TTA for threshold search and final evaluation")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Download dataset if requested
    if args.download:
        args.data_dir = download_dataset(args.data_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    use_indices = not args.no_indices
    in_channels = 18 if use_indices else 14

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_indices=use_indices,
        seed=args.seed,
    )

    # Model
    model, model_name = build_model(
        model_type=args.model_type,
        in_channels=in_channels,
        encoder=args.encoder,
        pretrained=True,
        boundary_aux=args.boundary_aux,
    )
    model = model.to(device)

    use_ema = args.use_ema or (args.consistency_weight > 0 and args.consistency_split != "none")
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # Loss
    base_seg_loss = CombinedLoss().to(device)
    if args.boundary_aux:
        criterion = BoundaryAwareLoss(seg_loss=base_seg_loss, boundary_weight=args.boundary_weight).to(device)
    else:
        criterion = base_seg_loss

    consistency_loader = None
    if args.consistency_weight > 0 and args.consistency_split != "none":
        consistency_loader = val_loader if args.consistency_split == "validation" else test_loader

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: OneCycleLR for super-convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Training loop
    best_val_f1 = 0.0
    best_val_iou = 0.0
    best_score = float("-inf")
    patience = 20
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}")
    print(f"Input channels: {in_channels} ({'with' if use_indices else 'without'} spectral indices)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            consistency_loader=consistency_loader,
            consistency_weight=args.consistency_weight,
            consistency_mode=args.consistency_mode,
            consistency_warmup_epochs=args.consistency_warmup_epochs,
            consistency_conf_low=args.consistency_conf_low,
            consistency_conf_high=args.consistency_conf_high,
            perturb_prob=args.consistency_drop_prob,
            perturb_max_drop=args.consistency_max_drop,
            perturb_noise_std=args.consistency_noise_std,
        )

        eval_model = ema_model if args.use_ema and ema_model is not None else model
        val_loss, val_metrics = validate(eval_model, val_loader, criterion, device, use_tta=False)

        # Step CosineAnnealing scheduler per epoch (if not using OneCycleLR per step)
        # scheduler.step()  # Not needed for OneCycleLR

        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(log)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_metrics['f1']*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} F1: {val_metrics['f1']*100:.1f}% "
            f"IoU: {val_metrics['iou']*100:.1f}% "
            f"P: {val_metrics['precision']*100:.1f}% R: {val_metrics['recall']*100:.1f}% | "
            f"{elapsed:.0f}s"
        )

        # Save best model (by F1)
        val_score = 0.6 * val_metrics["f1"] + 0.4 * val_metrics["iou"]  # Z-score
        if val_score > best_score:
            best_val_f1 = val_metrics["f1"]
            best_val_iou = val_metrics["iou"]
            best_score = val_score
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_iou": best_val_iou,
                "model_name": model_name,
                "in_channels": in_channels,
                "encoder": args.encoder,
                "model_type": args.model_type,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✓ New best! Z = {val_score*100:.2f}")
        else:
            patience_counter += 1

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(args.output_dir, "latest_model.pth"))

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best model and evaluate
    print(f"\n{'='*60}")
    print("Loading best model for final evaluation...")
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")
    latest_ckpt_path = os.path.join(args.output_dir, "latest_model.pth")

    if os.path.exists(best_ckpt_path):
        ckpt_path = best_ckpt_path
    elif os.path.exists(latest_ckpt_path):
        ckpt_path = latest_ckpt_path
        print("WARNING: best_model.pth not found. Falling back to latest_model.pth for evaluation.")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {args.output_dir}. Expected best_model.pth or latest_model.pth."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if args.use_ema and ema_model is not None and ckpt.get("ema_state_dict") is not None:
        ema_model.load_state_dict(ckpt["ema_state_dict"])

    eval_model = ema_model if args.use_ema and ema_model is not None else model

    # Find best threshold on val set
    best_thresh = find_best_threshold(eval_model, val_loader, device, use_tta=args.tta)

    # Evaluate on val set
    val_loss, val_metrics = validate(eval_model, val_loader, criterion, device, use_tta=args.tta)
    print(f"\nValidation Results (threshold={best_thresh:.2f}):")
    val_metrics_opt = compute_metrics(
        *_collect_predictions(eval_model, val_loader, device, use_tta=args.tta), threshold=best_thresh
    )
    print(f"  Precision: {val_metrics_opt['precision']*100:.2f}%")
    print(f"  Recall:    {val_metrics_opt['recall']*100:.2f}%")
    print(f"  F1 Score:  {val_metrics_opt['f1']*100:.2f}%")
    print(f"  IoU:       {val_metrics_opt['iou']*100:.2f}%")

    # Evaluate on test set
    print("\nTest Results:")
    test_metrics = evaluate_test(eval_model, test_loader, device, threshold=best_thresh, use_tta=args.tta)
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {test_metrics['f1']*100:.2f}%")
    print(f"  IoU:       {test_metrics['iou']*100:.2f}%")

    Z = 0.6 * test_metrics["f1"] * 100 + 0.4 * test_metrics["iou"] * 100
    print(f"\n  Z = 0.6 × {test_metrics['f1']*100:.2f} + 0.4 × {test_metrics['iou']*100:.2f} = {Z:.2f}")

    # Part A: Report trainable parameters
    print(f"\n{'='*60}")
    print(f"MODEL SUMMARY (Part A)")
    print(f"  Architecture: {model_name}")
    print(f"  Trainable Parameters: {model.count_parameters():,}")
    print(f"  Input Channels: {in_channels}")
    print(f"{'='*60}")

    # Save results
    results = {
        "model_name": model_name,
        "trainable_parameters": model.count_parameters(),
        "in_channels": in_channels,
        "best_threshold": best_thresh,
        "used_tta": args.tta,
        "used_ema": bool(args.use_ema),
        "used_boundary_aux": bool(args.boundary_aux),
        "consistency_weight": args.consistency_weight,
        "consistency_split": args.consistency_split,
        "val_metrics": val_metrics_opt,
        "test_metrics": test_metrics,
        "Z_score": Z,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


@torch.no_grad()
def _collect_predictions(model, loader, device, use_tta=False):
    model.eval()
    all_preds, all_targets = [], []
    for images, masks in loader:
        images = images.to(device)
        outputs = _forward_with_tta(model, images, use_tta=use_tta)
        all_preds.append(outputs.cpu())
        all_targets.append(masks)
    return torch.cat(all_preds), torch.cat(all_targets)


if __name__ == "__main__":
    main()

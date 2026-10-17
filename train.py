"""
train.py -- train the U-Net/ResNet34 crack segmenter.

Fixes vs the original: runs on GPU (.to(device)), ImageNet normalisation, fixed seed + fixed
split, augmentation, best-checkpoint saving, Dice + IoU logging to CSV, optional mixed precision.
NOT RUN in the authoring sandbox (no PyTorch there). Do a 1-epoch smoke test first:
    python train.py --images D/images --masks D/masks --epochs 1 --limit 200
Kaggle: turn Internet ON (pretrained weights download) and GPU ON.
Prints real numbers only from the real run; nothing here is pre-filled.
"""
import argparse, csv, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import DiceLoss

from model import build_model, get_device
from dataset import CrackDataset, pair_files


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ious, dices = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x)) > 0.5; t = y > 0.5
        inter = (p & t).sum((1, 2, 3)).float(); union = (p | t).sum((1, 2, 3)).float()
        both = p.sum((1, 2, 3)).float() + t.sum((1, 2, 3)).float()
        # images with no crack and no prediction count as perfect (1.0)
        ious.append(torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).mean().item())
        dices.append(torch.where(both > 0, 2 * inter / both.clamp(min=1), torch.ones_like(both)).mean().item())
    return float(np.mean(ious)), float(np.mean(dices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="use only N pairs (smoke test)")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--out", default="crack_unet_best.pth")
    ap.add_argument("--no-pretrained", action="store_true"); ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed); device = get_device(); print("device:", device)
    pairs, missing = pair_files(a.images, a.masks)
    print(f"{len(pairs)} matched pairs, {len(missing)} images without a mask")
    if not pairs: raise SystemExit("No matched image/mask pairs -- run check_dataset.py")
    rng = random.Random(a.seed); rng.shuffle(pairs)          # fixed, reproducible split
    if a.limit: pairs = pairs[:a.limit]
    n_val = max(1, int(a.val_frac * len(pairs)))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"train {len(train_pairs)} / val {len(val_pairs)} (seed {a.seed})")

    tl = DataLoader(CrackDataset(train_pairs, a.size, augment=True), a.batch, shuffle=True,
                    num_workers=a.workers, pin_memory=device.type == "cuda", drop_last=True)
    vl = DataLoader(CrackDataset(val_pairs, a.size), a.batch, shuffle=False, num_workers=a.workers)

    model = build_model(pretrained=not a.no_pretrained).to(device)
    dice, bce = DiceLoss(mode="binary"), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp and device.type == "cuda")

    best, log = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=a.amp and device.type == "cuda"):
                out = model(x); loss = dice(out, y) + bce(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        iou, dsc = evaluate(model, vl, device); sched.step(iou)
        row = {"epoch": ep, "train_loss": tot / max(1, len(tl)), "val_iou": iou, "val_dice": dsc}
        log.append(row); print(f"epoch {ep:2d}/{a.epochs}  loss {row['train_loss']:.4f}  val IoU {iou:.3f}  val Dice {dsc:.3f}")
        if iou > best:
            best = iou; torch.save(model.state_dict(), a.out); print(f"   saved best -> {a.out}")
    with open("training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    print(f"done. best val IoU {best:.3f}. Log: training_log.csv  (this is the log to quote, nothing else)")


if __name__ == "__main__":
    main()

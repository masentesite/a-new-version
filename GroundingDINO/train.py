"""Train and evaluate the local v2 multimodal GroundingDINO model.

Run from this directory:

    python train.py train
    python train.py test --checkpoint runs/v2/best.pth

The default paths are relative to this file, so the whole project can be
copied to a cloud machine without editing path strings.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_DIR / "../../data/TrainSet."
DEFAULT_WEIGHTS_ROOT = PROJECT_DIR / "../../weights"
DEFAULT_CONFIG = PROJECT_DIR / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_PRETRAINED = DEFAULT_WEIGHTS_ROOT / "groundingdino_swint_ogc.pth"
DEFAULT_TEXT_ENCODER = DEFAULT_WEIGHTS_ROOT / "bert-base-uncased"
DEFAULT_RUN_DIR = PROJECT_DIR / "runs/v2"

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
MODALITIES = ("visible", "infrared", "depth")


def _path(value: str | Path) -> Path:
    """Resolve a CLI path relative to this project, rather than the shell cwd."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_DIR / candidate).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _load_queries(data_root: Path) -> List[dict]:
    query_file = data_root / "queries/queries.json"
    if not query_file.is_file():
        raise FileNotFoundError(
            f"Cannot find {query_file}. Put TrainSet. beside the project data directory "
            "or pass --data-root."
        )

    raw = json.loads(query_file.read_text(encoding="utf-8"))
    records = []
    for qid, item in raw.items():
        bbox = item.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        records.append(
            {
                "qid": str(qid),
                "img_id": str(qid).rsplit("_", 1)[0],
                "visible": item["visible"],
                "infrared": item["infrared"],
                "depth": item["depth"],
                "query": str(item["query"]).strip(),
                "bbox": [float(v) for v in bbox],
            }
        )
    if not records:
        raise ValueError(f"No records with bounding boxes were found in {query_file}")
    return records


def _split_by_image(
    records: Sequence[dict], val_ratio: float, seed: int
) -> Tuple[List[dict], List[dict]]:
    image_ids = sorted({record["img_id"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * val_ratio)))
    val_ids = set(image_ids[:val_count])
    train_records = [record for record in records if record["img_id"] not in val_ids]
    val_records = [record for record in records if record["img_id"] in val_ids]
    return train_records, val_records


def _load_rgb(path: Path) -> Image.Image:
    # The backbone expects three channels. Converting auxiliary images to RGB
    # preserves their pixel values while keeping the v2 backbone interface.
    with Image.open(path) as image:
        return image.convert("RGB")


def _resize_modalities(
    images: Mapping[str, Image.Image], box: Sequence[float], size: int, max_size: int
) -> Tuple[Dict[str, Tensor], Tensor, Tuple[int, int]]:
    """Resize all modalities with exactly the same geometry."""
    reference = images["visible"]
    original_width, original_height = reference.size
    resized_visible, target = T.resize(
        reference,
        {"boxes": torch.tensor([box], dtype=torch.float32) * torch.tensor(
            [original_width, original_height, original_width, original_height]
        )},
        size,
        max_size,
    )

    transformed = {"visible": resized_visible}
    # T.resize accepts a (width, height) tuple and internally converts it to
    # the (height, width) form used by torchvision.
    target_size = (resized_visible.width, resized_visible.height)
    for modality in ("infrared", "depth"):
        transformed[modality], _ = T.resize(images[modality], None, target_size)

    tensors = {}
    for modality, image in transformed.items():
        tensor = T.ToTensor()(image, None)[0]
        tensor = T.Normalize(IMAGE_MEAN, IMAGE_STD)(tensor, None)[0]
        tensors[modality] = tensor

    resized_width, resized_height = transformed["visible"].size
    return tensors, target["boxes"][0], (resized_width, resized_height)


class TrainSetDataset(Dataset):
    """One sample per text query, with RGB, infrared, and depth inputs."""

    def __init__(
        self,
        records: Sequence[dict],
        data_root: Path,
        image_size: int = 800,
        max_size: int = 1333,
    ):
        self.records = list(records)
        self.data_root = data_root
        self.image_size = image_size
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        images = {
            modality: _load_rgb(self.data_root / record[modality])
            for modality in MODALITIES
        }
        tensors, box_xyxy, resized_size = _resize_modalities(
            images,
            record["bbox"],
            size=self.image_size,
            max_size=self.max_size,
        )
        width, height = resized_size
        normalized_box = box_xyxy / torch.tensor(
            [width, height, width, height], dtype=torch.float32
        )
        normalized_box = torch.stack(
            [
                (normalized_box[0] + normalized_box[2]) / 2,
                (normalized_box[1] + normalized_box[3]) / 2,
                normalized_box[2] - normalized_box[0],
                normalized_box[3] - normalized_box[1],
            ]
        ).clamp(0, 1)
        caption = record["query"].lower().strip()
        if not caption.endswith("."):
            caption += "."
        return {
            "samples": tensors,
            "caption": caption,
            "boxes": normalized_box,
            "qid": record["qid"],
            "img_id": record["img_id"],
            "original_size": [height, width],
        }


def _collate_batch(batch: Sequence[dict]) -> dict:
    samples = {
        modality: nested_tensor_from_tensor_list(
            [item["samples"][modality] for item in batch]
        )
        for modality in MODALITIES
    }
    return {
        "samples": samples,
        "captions": [item["caption"] for item in batch],
        "boxes": torch.stack([item["boxes"] for item in batch]),
        "qid": [item["qid"] for item in batch],
        "img_id": [item["img_id"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "samples": {
            modality: nested.to(device)
            for modality, nested in batch["samples"].items()
        },
        "boxes": batch["boxes"].to(device),
    }


def _positive_token_map(
    tokenizer, captions: Sequence[str], max_text_len: int, device
) -> Tensor:
    """Mark caption tokens as positive for the single described object."""
    tokenized = tokenizer(
        list(captions),
        padding="longest",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )
    positive = tokenized["attention_mask"].bool()
    for token_id in getattr(tokenizer, "all_special_ids", []):
        positive &= tokenized["input_ids"] != token_id
    if positive.shape[1] < max_text_len:
        positive = F.pad(positive, (0, max_text_len - positive.shape[1]), value=False)
    return positive.to(device)


class GroundingCriterion(nn.Module):
    """Single-target phrase grounding loss for TrainSet."""

    def __init__(
        self,
        tokenizer,
        max_text_len: int,
        box_loss_weight: float = 5.0,
        giou_loss_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.box_loss_weight = box_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _match(self, outputs: dict, boxes: Tensor, positive_map: Tensor) -> Tensor:
        logits = outputs["pred_logits"].sigmoid()
        pred_boxes = outputs["pred_boxes"]
        positive_score = logits.masked_fill(~positive_map[:, None, :], 0).amax(dim=-1)
        l1 = torch.cdist(pred_boxes, boxes[:, None, :], p=1).squeeze(-1)
        giou = torch.stack(
            [
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_boxes[index]),
                    box_cxcywh_to_xyxy(boxes[index : index + 1]),
                ).squeeze(-1)
                for index in range(pred_boxes.shape[0])
            ]
        )
        cost = l1 - 2.0 * positive_score - giou
        return cost.argmin(dim=1)

    def _loss_one(self, outputs: dict, boxes: Tensor, positive_map: Tensor) -> dict:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        batch_size, num_queries, text_len = pred_logits.shape
        matched = self._match(outputs, boxes, positive_map)

        target_logits = torch.zeros_like(pred_logits)
        target_logits[
            torch.arange(batch_size, device=pred_logits.device),
            matched,
        ] = positive_map.to(pred_logits.dtype)
        valid_logits = torch.isfinite(pred_logits)
        safe_logits = torch.where(valid_logits, pred_logits, torch.zeros_like(pred_logits))
        prob = safe_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(
            safe_logits, target_logits, reduction="none"
        )
        p_t = prob * target_logits + (1 - prob) * (1 - target_logits)
        alpha_t = self.focal_alpha * target_logits + (1 - self.focal_alpha) * (
            1 - target_logits
        )
        loss_ce = (
            alpha_t * ce * (1 - p_t).pow(self.focal_gamma) * valid_logits
        ).sum() / valid_logits.sum().clamp_min(1)

        selected_boxes = pred_boxes[
            torch.arange(batch_size, device=pred_boxes.device), matched
        ]
        loss_bbox = F.l1_loss(selected_boxes, boxes, reduction="none").sum(-1).mean()
        loss_giou = 1 - torch.stack(
            [
                generalized_box_iou(
                    box_cxcywh_to_xyxy(selected_boxes[index : index + 1]),
                    box_cxcywh_to_xyxy(boxes[index : index + 1]),
                ).squeeze()
                for index in range(batch_size)
            ]
        ).mean()
        total = loss_ce + self.box_loss_weight * loss_bbox + self.giou_loss_weight * loss_giou
        return {
            "loss": total,
            "loss_ce": loss_ce.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            "matched": matched.detach(),
        }

    def forward(self, outputs: dict, batch: dict) -> Tuple[Tensor, Dict[str, float]]:
        positive_map = _positive_token_map(
            self.tokenizer,
            batch["captions"],
            self.max_text_len,
            batch["boxes"].device,
        )
        main = self._loss_one(outputs, batch["boxes"], positive_map)
        total = main["loss"]
        metrics = {
            "loss": float(main["loss"].detach().cpu()),
            "loss_ce": float(main["loss_ce"].cpu()),
            "loss_bbox": float(main["loss_bbox"].cpu()),
            "loss_giou": float(main["loss_giou"].cpu()),
        }

        for auxiliary in outputs.get("aux_outputs", []):
            aux_loss = self._loss_one(auxiliary, batch["boxes"], positive_map)
            total = total + 0.5 * aux_loss["loss"]

        modality_outputs = outputs.get("aux_modality_outputs", {})
        aux_weight = float(outputs.get("aux_modality_loss_weight", 0.0))
        for auxiliary in modality_outputs.values():
            aux_loss = self._loss_one(auxiliary, batch["boxes"], positive_map)
            total = total + aux_weight * aux_loss["loss"]

        alignment_loss = outputs.get("fusion_alignment_loss")
        alignment_weight = float(outputs.get("alignment_loss_weight", 0.0))
        if alignment_loss is not None:
            total = total + alignment_weight * alignment_loss
            metrics["loss_alignment"] = float(alignment_loss.detach().cpu())

        metrics["loss"] = float(total.detach().cpu())
        return total, metrics


def _build_model(config_path: Path, device: torch.device, text_encoder: Optional[Path]):
    args = SLConfig.fromfile(str(config_path))
    if text_encoder is not None and text_encoder.is_dir():
        args.text_encoder_type = str(text_encoder)
    args.device = str(device)
    model = build_model(args)
    return model, args


def _load_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device, optimizer=None, scaler=None
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    result = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(
        f"Loaded {checkpoint_path}: missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    metrics: dict,
    args: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "args": args,
        },
        path,
    )


def _run_epoch(
    model: nn.Module,
    criterion: GroundingCriterion,
    loader: DataLoader,
    device: torch.device,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    log_interval: int = 20,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    count = 0
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp and device.type == "cuda"):
                outputs = model(
                    batch["samples"],
                    captions=batch["captions"],
                    unset_image_tensor=True,
                )
                loss, metrics = criterion(outputs, batch)
            if training:
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
            batch_size = len(batch["captions"])
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            if step % log_interval == 0 or step == len(loader):
                prefix = "train" if training else "test"
                print(
                    f"[{prefix}] {step:4d}/{len(loader)} "
                    + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                )
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def _evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tokenizer,
    max_text_len: int,
    amp: bool,
) -> dict:
    model.eval()
    ious = []
    results = []
    for batch in loader:
        batch = _move_batch(batch, device)
        positive_map = _positive_token_map(
            tokenizer, batch["captions"], max_text_len, device
        )
        with autocast(enabled=amp and device.type == "cuda"):
            outputs = model(
                batch["samples"],
                captions=batch["captions"],
                unset_image_tensor=True,
            )
        scores = outputs["pred_logits"].sigmoid().masked_fill(
            ~positive_map[:, None, :], 0
        ).amax(dim=-1)
        best = scores.argmax(dim=1)
        pred_boxes = outputs["pred_boxes"][
            torch.arange(len(best), device=device), best
        ]
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(batch["boxes"])
        batch_iou = torch.diag(
            _pairwise_iou(pred_xyxy, gt_xyxy)
        ).detach().cpu().tolist()
        ious.extend(batch_iou)
        for index, iou in enumerate(batch_iou):
            results.append(
                {
                    "qid": batch["qid"][index],
                    "img_id": batch["img_id"][index],
                    "query": batch["captions"][index],
                    "score": float(scores[index, best[index]].cpu()),
                    "pred_box": pred_boxes[index].cpu().tolist(),
                    "gt_box": batch["boxes"][index].cpu().tolist(),
                    "iou": float(iou),
                }
            )
    values = np.asarray(ious, dtype=np.float32)
    return {
        "num_queries": int(values.size),
        "mean_iou": float(values.mean()) if values.size else 0.0,
        "median_iou": float(np.median(values)) if values.size else 0.0,
        "acc@0.25": float((values >= 0.25).mean()) if values.size else 0.0,
        "acc@0.5": float((values >= 0.50).mean()) if values.size else 0.0,
        "acc@0.75": float((values >= 0.75).mean()) if values.size else 0.0,
        "results": results,
    }


def _pairwise_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0)
    intersection = intersection[..., 0] * intersection[..., 1]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp(min=0)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-6)


def train(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_CONFIG,
    pretrained: str | Path | None = DEFAULT_PRETRAINED,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    epochs: int = 20,
    batch_size: int = 1,
    workers: int = 2,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.1,
    image_size: int = 800,
    max_size: int = 1333,
    seed: int = 42,
    device: str = "auto",
    amp: bool = True,
    resume: str | Path | None = None,
) -> dict:
    """Train v2 and save resumable checkpoints under ``run_dir``."""
    data_root = _path(data_root)
    config_path = _path(config_path)
    run_dir = _path(run_dir)
    pretrained_path = _path(pretrained) if pretrained else None
    resume_path = _path(resume) if resume else None
    _seed_everything(seed)

    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    records = _load_queries(data_root)
    train_records, val_records = _split_by_image(records, val_ratio, seed)
    text_encoder = DEFAULT_TEXT_ENCODER if DEFAULT_TEXT_ENCODER.is_dir() else None
    model, model_args = _build_model(config_path, device_obj, text_encoder)
    model.to(device_obj)

    if pretrained_path is not None and pretrained_path.is_file():
        _load_checkpoint(model, pretrained_path, device_obj)
    elif pretrained_path is not None:
        print(f"Pretrained checkpoint not found, training from scratch: {pretrained_path}")

    train_set = TrainSetDataset(train_records, data_root, image_size, max_size)
    val_set = TrainSetDataset(val_records, data_root, image_size, max_size)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device_obj.type == "cuda",
        "collate_fn": _collate_batch,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    criterion = GroundingCriterion(model.tokenizer, model_args.max_text_len).to(device_obj)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = GradScaler(enabled=amp and device_obj.type == "cuda")

    start_epoch = 0
    if resume_path is not None:
        checkpoint = _load_checkpoint(
            model, resume_path, device_obj, optimizer=optimizer, scaler=scaler
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        for _ in range(start_epoch):
            scheduler.step()

    run_dir.mkdir(parents=True, exist_ok=True)
    config_dump = {
        "data_root": data_root,
        "config_path": config_path,
        "pretrained": pretrained_path,
        "run_dir": run_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_ratio": val_ratio,
        "image_size": image_size,
        "max_size": max_size,
        "seed": seed,
        "device": str(device_obj),
        "amp": amp,
        "train_queries": len(train_set),
        "val_queries": len(val_set),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_dump, indent=2, default=_json_default), encoding="utf-8"
    )

    history = []
    best_iou = -math.inf
    for epoch in range(start_epoch, epochs):
        started = time.time()
        print(
            f"\nEpoch {epoch + 1}/{epochs} | train={len(train_set)} | "
            f"val={len(val_set)} | device={device_obj}"
        )
        train_metrics = _run_epoch(
            model, criterion, train_loader, device_obj, optimizer, scaler, amp
        )
        val_metrics = _run_epoch(model, criterion, val_loader, device_obj, amp=amp)
        prediction_metrics = _evaluate_predictions(
            model, val_loader, device_obj, model.tokenizer, model_args.max_text_len, amp
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "val_predictions": {
                key: value for key, value in prediction_metrics.items() if key != "results"
            },
        }
        history.append(record)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=_json_default), encoding="utf-8"
        )
        _save_checkpoint(
            run_dir / "last.pth",
            model,
            optimizer,
            scaler,
            epoch,
            record,
            config_dump,
        )
        (run_dir / "val_predictions.json").write_text(
            json.dumps(prediction_metrics, indent=2, default=_json_default), encoding="utf-8"
        )
        if prediction_metrics["mean_iou"] > best_iou:
            best_iou = prediction_metrics["mean_iou"]
            _save_checkpoint(
                run_dir / "best.pth",
                model,
                optimizer,
                scaler,
                epoch,
                record,
                config_dump,
            )
            print(f"Saved new best checkpoint, val mean IoU={best_iou:.4f}")

    return {"run_dir": str(run_dir), "best_val_mean_iou": best_iou, "history": history}


def test(
    checkpoint: str | Path,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config_path: str | Path = DEFAULT_CONFIG,
    workers: int = 2,
    batch_size: int = 1,
    val_ratio: float = 0.1,
    image_size: int = 800,
    max_size: int = 1333,
    seed: int = 42,
    device: str = "auto",
    output: str | Path | None = None,
    all_data: bool = False,
    amp: bool = True,
) -> dict:
    """Evaluate a checkpoint and write per-query IoU results."""
    data_root = _path(data_root)
    config_path = _path(config_path)
    checkpoint_path = _path(checkpoint)
    if output is None:
        output_path = checkpoint_path.parent / "test_results.json"
    else:
        output_path = _path(output)
    _seed_everything(seed)
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    records = _load_queries(data_root)
    if all_data:
        test_records = records
    else:
        _, test_records = _split_by_image(records, val_ratio, seed)
    text_encoder = DEFAULT_TEXT_ENCODER if DEFAULT_TEXT_ENCODER.is_dir() else None
    model, model_args = _build_model(config_path, device_obj, text_encoder)
    model.to(device_obj)
    _load_checkpoint(model, checkpoint_path, device_obj)

    dataset = TrainSetDataset(test_records, data_root, image_size, max_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device_obj.type == "cuda",
        collate_fn=_collate_batch,
        persistent_workers=workers > 0,
    )
    metrics = _evaluate_predictions(
        model, loader, device_obj, model.tokenizer, model_args.max_text_len, amp
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    print(
        f"Tested {metrics['num_queries']} queries | "
        f"mean IoU={metrics['mean_iou']:.4f} | acc@0.5={metrics['acc@0.5']:.4f}"
    )
    print(f"Results: {output_path}")
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test GroundingDINO v2 on TrainSet.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and save checkpoints")
    train_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    train_parser.add_argument("--config", dest="config_path", default=str(DEFAULT_CONFIG))
    train_parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    train_parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--val-ratio", type=float, default=0.1)
    train_parser.add_argument("--image-size", type=int, default=800)
    train_parser.add_argument("--max-size", type=int, default=1333)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--no-amp", action="store_true")

    test_parser = subparsers.add_parser("test", help="evaluate a saved checkpoint")
    test_parser.add_argument("--checkpoint", default=str(DEFAULT_RUN_DIR / "best.pth"))
    test_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    test_parser.add_argument("--config", dest="config_path", default=str(DEFAULT_CONFIG))
    test_parser.add_argument("--workers", type=int, default=2)
    test_parser.add_argument("--batch-size", type=int, default=1)
    test_parser.add_argument("--val-ratio", type=float, default=0.1)
    test_parser.add_argument("--image-size", type=int, default=800)
    test_parser.add_argument("--max-size", type=int, default=1333)
    test_parser.add_argument("--seed", type=int, default=42)
    test_parser.add_argument("--device", default="auto")
    test_parser.add_argument("--output", default=None)
    test_parser.add_argument("--all-data", action="store_true")
    test_parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "train":
        train(
            data_root=args.data_root,
            config_path=args.config_path,
            pretrained=args.pretrained,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_ratio=args.val_ratio,
            image_size=args.image_size,
            max_size=args.max_size,
            seed=args.seed,
            device=args.device,
            amp=not args.no_amp,
            resume=args.resume,
        )
    else:
        test(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            config_path=args.config_path,
            workers=args.workers,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            image_size=args.image_size,
            max_size=args.max_size,
            seed=args.seed,
            device=args.device,
            output=args.output,
            all_data=args.all_data,
            amp=not args.no_amp,
        )


if __name__ == "__main__":
    main()

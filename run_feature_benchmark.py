#!/usr/bin/env python3
"""Fair frozen-feature benchmark for small dental X-ray datasets.

The script compares four encoder families under one protocol:
  * ResNet18
  * DINOv3 ViT-S/16
  * SigLIP2 Base
  * MambaVision-T

Every backbone is frozen. Images are embedded once, then each cross-validation
fold uses StandardScaler -> PCA -> class-balanced LogisticRegression. PCA is
fit only on the training portion of a fold, preventing validation leakage.

Expected directory layout:

    DATA_DIR/
      healthy/
        image_001.jpg
      caries/
        image_002.jpg

Class directories may contain nested folders. Alternatively, pass a CSV with
--manifest. It must contain image_path and label columns; paths may be absolute
or relative to --data-dir. An optional group column can keep related images in
the same fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODEL_NAMES = ("resnet18", "dinov3_vits16", "siglip2", "mambavision_t")
DEFAULT_MODEL_NAMES = ("resnet18", "dinov3_vits16", "siglip2")


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    manifest: str | None
    output_dir: str
    models: list[str]
    classes: list[str] | None
    folds: int
    seed: int
    image_size: int
    pca_dim: int
    batch_size: int
    num_workers: int
    device: str
    group_column: str | None
    force_features: bool


class ImagePathDataset(Dataset):
    def __init__(self, paths: Iterable[str], transform: Callable[[Image.Image], object]):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), index


class ProcessorCollator:
    """Collate PIL images with a Hugging Face image processor."""

    def __init__(self, processor, image_size: int):
        self.processor = processor
        self.common_geometry = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Grayscale(num_output_channels=3),
            ]
        )

    def __call__(self, batch):
        images = []
        indices = []
        for image, index in batch:
            images.append(self.common_geometry(image))
            indices.append(index)
        encoded = self.processor(images=images, return_tensors="pt")
        return encoded, torch.tensor(indices, dtype=torch.long)


class PILPathDataset(Dataset):
    def __init__(self, paths: Iterable[str]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return image.convert("RGB"), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(DEFAULT_MODEL_NAMES),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Optional class-directory names to include, e.g. healthy caries.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--group-column",
        help="Optional manifest column, such as patient_id or original_image_id.",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Ignore cached embeddings and recompute them.",
    )
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scan_image_folders(data_dir: Path, classes: list[str] | None) -> pd.DataFrame:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if classes:
        class_dirs = [data_dir / name for name in classes]
        missing = [str(path) for path in class_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "The following class directories are missing:\n  " + "\n  ".join(missing)
            )
    else:
        class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())

    if len(class_dirs) < 2:
        raise ValueError(
            "Need at least two class directories. Expected DATA_DIR/class_name/images."
        )

    rows = []
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                rows.append(
                    {
                        "image_path": str(path.resolve()),
                        "label": class_dir.name,
                    }
                )
    if not rows:
        raise ValueError(f"No supported images found below {data_dir}")
    return pd.DataFrame(rows)


def load_manifest(
    data_dir: Path,
    manifest: Path,
    classes: list[str] | None,
) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    def resolve_path(value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = data_dir / path
        return str(path.resolve())

    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str).map(resolve_path)
    frame["label"] = frame["label"].astype(str)
    if classes:
        frame = frame[frame["label"].isin(classes)].copy()
    nonexistent = [path for path in frame["image_path"] if not Path(path).is_file()]
    if nonexistent:
        preview = "\n  ".join(nonexistent[:10])
        raise FileNotFoundError(f"Manifest contains missing images:\n  {preview}")
    return frame.reset_index(drop=True)


def assign_labels_and_folds(
    frame: pd.DataFrame,
    folds: int,
    seed: int,
    group_column: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.copy().sort_values("image_path").reset_index(drop=True)
    class_names = sorted(frame["label"].unique().tolist())
    class_to_id = {name: index for index, name in enumerate(class_names)}
    frame["label_id"] = frame["label"].map(class_to_id).astype(int)

    counts = frame["label"].value_counts()
    if counts.min() < folds:
        raise ValueError(
            f"Smallest class has {counts.min()} images, fewer than --folds={folds}. "
            "Reduce the number of folds."
        )

    frame["fold"] = -1
    if group_column:
        if group_column not in frame.columns:
            raise ValueError(f"Group column not found in manifest: {group_column}")
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(
            frame["image_path"], frame["label_id"], groups=frame[group_column]
        )
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(frame["image_path"], frame["label_id"])

    for fold_index, (_, validation_indices) in enumerate(split_iter):
        frame.loc[validation_indices, "fold"] = fold_index
    return frame, class_to_id


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["image_path", "label_id"]].itertuples(index=False):
        path = Path(row.image_path)
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(row.label_id).encode())
    return digest.hexdigest()


def image_net_transform(image_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_resnet(device: torch.device, image_size: int):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    transform = image_net_transform(
        image_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    def forward(inputs):
        return model(inputs)

    return model, ImagePathDataset, transform, None, forward


def load_hf_encoder(model_id: str, device: torch.device):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("Install transformers from requirements.txt") from error

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval().to(device)
    return model, processor


def pooled_hf_output(outputs: object) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)):
        candidate = outputs[0]
        if isinstance(candidate, torch.Tensor):
            if candidate.ndim == 3:
                return candidate[:, 0]
            return candidate
    pooler = getattr(outputs, "pooler_output", None)
    if isinstance(pooler, torch.Tensor):
        return pooler
    image_embeds = getattr(outputs, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden[:, 0]
    raise TypeError(f"Could not locate a pooled embedding in output type {type(outputs)}")


def load_dinov3(device: torch.device, image_size: int):
    model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
    model, processor = load_hf_encoder(model_id, device)

    def forward(encoded):
        outputs = model(**encoded)
        return pooled_hf_output(outputs)

    return model, PILPathDataset, None, ProcessorCollator(processor, image_size), forward


def load_siglip2(device: torch.device, image_size: int):
    model_id = "google/siglip2-base-patch16-224"
    model, processor = load_hf_encoder(model_id, device)

    def forward(encoded):
        pixel_values = encoded["pixel_values"]
        if hasattr(model, "get_image_features"):
            result = model.get_image_features(pixel_values=pixel_values)
            if isinstance(result, torch.Tensor):
                return result
            return pooled_hf_output(result)
        outputs = model.vision_model(pixel_values=pixel_values)
        return pooled_hf_output(outputs)

    return model, PILPathDataset, None, ProcessorCollator(processor, image_size), forward


def load_mambavision(device: torch.device, image_size: int):
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "MambaVision requires the optional mamba-ssm extension. Install it "
            "after PyTorch with: `pip install packaging ninja` followed by "
            "`pip install mamba-ssm==2.3.2.post1 --no-build-isolation`. "
            "Do not install the mambavision PyPI package."
        ) from error
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError("Install transformers from requirements.txt") from error

    model = AutoModel.from_pretrained(
        "nvidia/MambaVision-T-1K",
        trust_remote_code=True,
    )
    model.eval().to(device)
    mean = getattr(model.config, "mean", (0.485, 0.456, 0.406))
    std = getattr(model.config, "std", (0.229, 0.224, 0.225))
    transform = image_net_transform(image_size, mean, std)

    def forward(inputs):
        outputs = model(inputs)
        return pooled_hf_output(outputs)

    return model, ImagePathDataset, transform, None, forward


MODEL_LOADERS = {
    "resnet18": load_resnet,
    "dinov3_vits16": load_dinov3,
    "siglip2": load_siglip2,
    "mambavision_t": load_mambavision,
}


def move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    if hasattr(batch, "to"):
        return batch.to(device)
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def extract_features(
    model_name: str,
    paths: list[str],
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader_function = MODEL_LOADERS[model_name]
    model, dataset_class, transform, collator, forward = loader_function(device, image_size)
    if dataset_class is PILPathDataset:
        dataset = dataset_class(paths)
    else:
        dataset = dataset_class(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    feature_rows: list[np.ndarray | None] = [None] * len(dataset)
    with torch.inference_mode():
        for batch, indices in tqdm(loader, desc=f"Embedding {model_name}"):
            batch = move_batch_to_device(batch, device)
            embeddings = forward(batch)
            embeddings = embeddings.detach().float().cpu().numpy()
            if embeddings.ndim > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            for embedding, index in zip(embeddings, indices.tolist()):
                feature_rows[index] = embedding

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if any(row is None for row in feature_rows):
        raise RuntimeError(f"Missing extracted features for {model_name}")
    return np.stack(feature_rows).astype(np.float32)


def multiclass_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def evaluate_model(
    model_name: str,
    features: np.ndarray,
    frame: pd.DataFrame,
    folds: int,
    requested_pca_dim: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["label_id"].to_numpy()
    fold_rows = []
    prediction_rows = []

    for fold in range(folds):
        train_mask = frame["fold"].to_numpy() != fold
        validation_mask = ~train_mask
        x_train = features[train_mask]
        x_validation = features[validation_mask]
        y_train = labels[train_mask]
        y_validation = labels[validation_mask]

        pca_dim = min(requested_pca_dim, x_train.shape[0] - 1, x_train.shape[1])
        if pca_dim < 1:
            raise ValueError("Not enough training samples for PCA")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)

        pca = PCA(n_components=pca_dim, whiten=True, random_state=seed)
        x_train = pca.fit_transform(x_train)
        x_validation = pca.transform(x_validation)

        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear" if len(np.unique(labels)) == 2 else "lbfgs",
            max_iter=2000,
            random_state=seed,
        )
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_validation)
        probabilities = classifier.predict_proba(x_validation)

        metrics = {
            "model": model_name,
            "fold": fold,
            "n_train": int(train_mask.sum()),
            "n_validation": int(validation_mask.sum()),
            "raw_feature_dim": int(features.shape[1]),
            "pca_dim": int(pca_dim),
            "accuracy": accuracy_score(y_validation, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_validation, predictions),
            "precision_macro": precision_score(
                y_validation, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_validation, predictions, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_validation, predictions, average="macro", zero_division=0
            ),
            "auroc": multiclass_auc(y_validation, probabilities),
        }
        fold_rows.append(metrics)

        validation_indices = np.flatnonzero(validation_mask)
        for local_index, row_index in enumerate(validation_indices):
            row = {
                "model": model_name,
                "fold": fold,
                "image_path": frame.iloc[row_index]["image_path"],
                "label": frame.iloc[row_index]["label"],
                "label_id": int(y_validation[local_index]),
                "prediction_id": int(predictions[local_index]),
            }
            for class_index, probability in enumerate(probabilities[local_index]):
                row[f"probability_class_{class_index}"] = float(probability)
            prediction_rows.append(row)

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auroc",
    ]
    rows = []
    for model_name, group in fold_metrics.groupby("model", sort=False):
        row = {"model": model_name}
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)


def main() -> int:
    args = parse_args()
    set_deterministic(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)

    if args.manifest:
        frame = load_manifest(args.data_dir, args.manifest, args.classes)
    else:
        frame = scan_image_folders(args.data_dir, args.classes)
    frame, class_to_id = assign_labels_and_folds(
        frame, args.folds, args.seed, args.group_column
    )
    frame.to_csv(args.output_dir / "splits.csv", index=False)
    fingerprint = dataset_fingerprint(frame)

    config = RunConfig(
        data_dir=str(args.data_dir.resolve()),
        manifest=str(args.manifest.resolve()) if args.manifest else None,
        output_dir=str(args.output_dir.resolve()),
        models=args.models,
        classes=args.classes,
        folds=args.folds,
        seed=args.seed,
        image_size=args.image_size,
        pca_dim=args.pca_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        group_column=args.group_column,
        force_features=args.force_features,
    )
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                **asdict(config),
                "class_to_id": class_to_id,
                "dataset_fingerprint": fingerprint,
                "torch_version": torch.__version__,
            },
            handle,
            indent=2,
        )

    print(f"Device: {device}")
    print(f"Images: {len(frame)}")
    print(f"Classes: {class_to_id}")
    print(f"Models: {', '.join(args.models)}")

    all_fold_metrics = []
    all_predictions = []
    paths = frame["image_path"].tolist()

    for model_name in args.models:
        cache_path = feature_dir / f"{model_name}.npz"
        features = None
        if cache_path.exists() and not args.force_features:
            cache = np.load(cache_path, allow_pickle=False)
            cached_fingerprint = str(cache["dataset_fingerprint"].item())
            if cached_fingerprint == fingerprint:
                features = cache["features"]
                print(f"Using cached features: {cache_path}")
            else:
                print(f"Ignoring stale feature cache: {cache_path}")

        if features is None:
            start = time.time()
            try:
                features = extract_features(
                    model_name=model_name,
                    paths=paths,
                    device=device,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            except Exception as error:
                print(f"\nFailed while loading or running {model_name}: {error}", file=sys.stderr)
                if model_name == "dinov3_vits16":
                    print(
                        "DINOv3 is gated. Accept its Hugging Face license and run "
                        "`hf auth login` before retrying.",
                        file=sys.stderr,
                    )
                return 2
            np.savez_compressed(
                cache_path,
                features=features,
                dataset_fingerprint=np.array(fingerprint),
            )
            print(
                f"Saved {features.shape} features to {cache_path} "
                f"in {time.time() - start:.1f}s"
            )

        fold_metrics, predictions = evaluate_model(
            model_name=model_name,
            features=features,
            frame=frame,
            folds=args.folds,
            requested_pca_dim=args.pca_dim,
            seed=args.seed,
        )
        all_fold_metrics.append(fold_metrics)
        all_predictions.append(predictions)

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_metrics(fold_metrics)

    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print("\nSummary (mean across folds):")
    print(
        summary[
            ["model", "balanced_accuracy_mean", "f1_macro_mean", "auroc_mean"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nResults written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


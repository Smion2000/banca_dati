"""Tools to train an MLP on bankruptcy datasets.

This script loads Excel workbooks from the ``tesi`` directory and trains a
multi-layer perceptron (MLP) with the requested architecture.  Each sheet in
every workbook is treated as an independent sample.  The features correspond to
values in the range A1:X11 (240 cells) while the targets are AA11, AB11 and AC11
respectively representing direct costs, indirect costs and the probability of
failure.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


EXPECTED_FEATURE_ROWS = 11
EXPECTED_FEATURE_COLS = 24  # columns A-X inclusive
COST_COLUMN_INDICES = (26, 27)  # AA, AB (0-based)
PROBABILITY_COLUMN_INDEX = 28  # AC (0-based)
FEATURE_SIZE = EXPECTED_FEATURE_ROWS * EXPECTED_FEATURE_COLS


@dataclass
class BankruptcyDataset(Dataset):
    """Torch dataset wrapping feature and target arrays."""

    features: torch.Tensor
    targets: torch.Tensor

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class BankruptcyMLP(nn.Module):
    """Multi-layer perceptron with the architecture required by the user."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_SIZE, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 25),
            nn.ReLU(),
            nn.Linear(25, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def collect_workbooks(data_root: Path) -> List[Path]:
    """Collect Excel workbooks from the two required sub-directories."""

    workbooks: List[Path] = []
    for sub in ("Attive_complete", "Fallite_complete"):
        folder = data_root / sub
        if not folder.exists():
            raise FileNotFoundError(f"Missing expected directory: {folder}")
        workbooks.extend(sorted(folder.glob("*.xlsx")))
    if not workbooks:
        raise FileNotFoundError(
            "No Excel files were found in the expected directories."
        )
    return workbooks


def extract_samples(workbook: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and targets from every worksheet in an Excel file."""

    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    try:
        sheets = pd.read_excel(workbook, sheet_name=None, header=None)
    except ValueError as exc:  # pragma: no cover - only triggered on malformed file
        raise ValueError(f"Failed to read '{workbook}': {exc}") from exc

    for sheet_name, sheet in sheets.items():
        if sheet.shape[0] < EXPECTED_FEATURE_ROWS or sheet.shape[1] <= PROBABILITY_COLUMN_INDEX:
            raise ValueError(
                f"Sheet '{sheet_name}' in '{workbook}' does not contain the expected number "
                "of rows/columns."
            )

        block = sheet.iloc[:EXPECTED_FEATURE_ROWS, :EXPECTED_FEATURE_COLS].to_numpy(dtype=float)
        feature_vector = block.flatten(order="C")

        output_values = sheet.iloc[
            EXPECTED_FEATURE_ROWS - 1,
            list(COST_COLUMN_INDICES) + [PROBABILITY_COLUMN_INDEX],
        ].to_numpy(dtype=float)

        features.append(feature_vector)
        targets.append(output_values)

    return np.vstack(features), np.vstack(targets)


def load_dataset(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load every workbook and concatenate the resulting samples."""

    all_features: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for workbook in collect_workbooks(data_root):
        f, t = extract_samples(workbook)
        all_features.append(f)
        all_targets.append(t)

    return np.vstack(all_features), np.vstack(all_targets)


def prepare_tensors(
    features: np.ndarray,
    targets: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler, StandardScaler]:
    """Scale the data and convert them into tensors."""

    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)

    cost_scaler = StandardScaler()
    scaled_costs = cost_scaler.fit_transform(targets[:, :2])
    scaled_targets = np.concatenate([scaled_costs, targets[:, 2:3]], axis=1)

    features_tensor = torch.from_numpy(scaled_features).float()
    targets_tensor = torch.from_numpy(scaled_targets).float()

    return features_tensor, targets_tensor, feature_scaler, cost_scaler


def split_dataset(
    features_tensor: torch.Tensor,
    targets_tensor: torch.Tensor,
    test_size: float,
    random_state: int,
) -> Tuple[BankruptcyDataset, BankruptcyDataset]:
    X_train, X_test, y_train, y_test = train_test_split(
        features_tensor.numpy(),
        targets_tensor.numpy(),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=None,
    )

    train_dataset = BankruptcyDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_dataset = BankruptcyDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    return train_dataset, test_dataset


def train_model(
    model: BankruptcyMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    train_losses: List[float] = []
    test_losses: List[float] = []

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            cost_pred = outputs[:, :2]
            prob_logit = outputs[:, 2]

            loss_cost = mse_loss(cost_pred, batch_targets[:, :2])
            loss_prob = bce_loss(prob_logit, batch_targets[:, 2])
            loss = loss_cost + loss_prob

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch_features.size(0)

        average_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(average_train_loss)

        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_features)
                cost_pred = outputs[:, :2]
                prob_logit = outputs[:, 2]

                loss_cost = mse_loss(cost_pred, batch_targets[:, :2])
                loss_prob = bce_loss(prob_logit, batch_targets[:, 2])
                loss = loss_cost + loss_prob

                epoch_test_loss += loss.item() * batch_features.size(0)

        average_test_loss = epoch_test_loss / len(test_loader.dataset)
        test_losses.append(average_test_loss)

        print(
            f"Epoch {epoch:03d} | Train Loss: {average_train_loss:.6f} | "
            f"Validation Loss: {average_test_loss:.6f}"
        )

    return train_losses, test_losses


def evaluate_model(
    model: BankruptcyMLP,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[float] = []
    y_pred_prob: List[float] = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)

            prob_pred = torch.sigmoid(outputs[:, 2]).cpu().numpy()

            y_true.extend(targets[:, 2].cpu().numpy().tolist())
            y_pred_prob.extend(prob_pred.tolist())

    y_true_array = np.array(y_true)
    y_pred_prob_array = np.array(y_pred_prob)
    y_pred_class = (y_pred_prob_array >= threshold).astype(int)
    y_true_class = (y_true_array >= threshold).astype(int)

    return y_true_class, y_pred_class, y_pred_prob_array


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Non-failure", "Failure"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Failure Probability)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_scalers(
    output_dir: Path,
    feature_scaler: StandardScaler,
    cost_scaler: StandardScaler,
) -> Path:
    scaler_path = output_dir / "scalers.npz"
    np.savez(
        scaler_path,
        feature_mean=feature_scaler.mean_,
        feature_scale=feature_scaler.scale_,
        cost_mean=cost_scaler.mean_,
        cost_scale=cost_scaler.scale_,
    )
    return scaler_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on the bankruptcy dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("tesi"),
        help="Path to the 'tesi' directory containing the Attive_complete and Fallite_complete folders.",
    )
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Mount Google Drive via google.colab before loading the dataset.",
    )
    parser.add_argument(
        "--drive-mount-point",
        type=Path,
        default=Path("/content/drive"),
        help="Mount point used when --mount-drive is supplied.",
    )
    parser.add_argument(
        "--drive-data-path",
        type=Path,
        default=Path("MyDrive/tesi"),
        help=(
            "Relative path inside the mounted Google Drive that contains the 'tesi' folder."
            " Ignored unless --mount-drive is used."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation/testing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert failure probabilities into binary classes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where artefacts such as the confusion matrix are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' selects CUDA when available.",
    )
    return parser.parse_args(argv)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def mount_google_drive(mount_point: Path) -> Path:
    """Mount Google Drive in Colab environments and return the mount point."""

    try:
        from google.colab import drive as colab_drive  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires Colab runtime
        raise RuntimeError(
            "The google.colab package is not available. --mount-drive can only be used inside Colab."
        ) from exc

    colab_drive.mount(str(mount_point), force_remount=False)
    return mount_point


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)

    if args.mount_drive:
        drive_root = mount_google_drive(args.drive_mount_point)
        data_root = drive_root / args.drive_data_path
    else:
        data_root = args.data_root

    features, targets = load_dataset(data_root)
    features_tensor, targets_tensor, feature_scaler, cost_scaler = prepare_tensors(features, targets)

    train_dataset, test_dataset = split_dataset(
        features_tensor, targets_tensor, test_size=args.test_size, random_state=42
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = BankruptcyMLP()
    train_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    y_true_class, y_pred_class, _ = evaluate_model(
        model,
        test_loader,
        device=device,
        threshold=args.threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = args.output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true_class, y_pred_class, confusion_path)

    scaler_path = save_scalers(args.output_dir, feature_scaler, cost_scaler)

    summary = {
        "threshold": args.threshold,
        "confusion_matrix_png": str(confusion_path.resolve()),
        "scalers_npz": str(scaler_path.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

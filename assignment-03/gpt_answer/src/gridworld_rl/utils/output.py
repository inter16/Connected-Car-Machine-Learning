"""Four-by-four terminal, CSV, text, and PNG output for screenshots."""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def format_value_grid(values: torch.Tensor, size: int = 4) -> str:
    if values.numel() != size * size:
        raise ValueError(f"Expected exactly {size * size} entries")
    matrix = values.detach().cpu().reshape(size, size)
    lines = []
    for row in matrix:
        lines.append("  ".join(f"{float(x.item()):9.3f}" for x in row))
    return "\n".join(lines)


def save_value_csv(values: torch.Tensor, filename: Path, size: int = 4) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    matrix = values.detach().cpu().reshape(size, size)
    with filename.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"col_{j}" for j in range(size)])
        writer.writerows([[f"{float(x.item()):.6f}" for x in row] for row in matrix])


def save_value_image(values: torch.Tensor, filename: Path, title: str) -> None:
    # Import lazily: PyTorch algorithms can be run independently of plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = values.detach().cpu().reshape(4, 4)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    heatmap = ax.imshow(matrix, cmap="viridis")
    for row in range(4):
        for col in range(4):
            state = row * 4 + col
            label = f"S{state}\n{float(matrix[row, col].item()):.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=11,
                    color="white" if matrix[row, col].item() < -20 else "black")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    fig.colorbar(heatmap, ax=ax, label="Estimated value V(s)")
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)

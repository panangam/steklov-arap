from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from steklov_arap.arap import cotan_laplacian_robust, load_mesh
from steklov_arap.steklov import K, load_cached_steklov_dtn


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = SCRIPT_DIR.parent / "meshes" / "sphere.obj"


def steklov_matrix(V: torch.Tensor, F: torch.Tensor, *, interior: bool, modes: int) -> torch.Tensor:
    cached_dtn = load_cached_steklov_dtn(V, F, interior=interior, K=modes)

    steklov_evals = cached_dtn["evals_int"].to(dtype=V.dtype, device=V.device)
    steklov_evecs = cached_dtn["evecs_int"].to(dtype=V.dtype, device=V.device)
    mass = cached_dtn["mass"].to(dtype=V.dtype, device=V.device)

    steklov_evals[0] = 0
    return (mass[:, None] * steklov_evecs * torch.pow(steklov_evals[None, :], 2)) @ (
        steklov_evecs.mT * mass[None, :]
    )


def save_matrix_visualization(
    matrix: torch.Tensor,
    output_path: Path,
    *,
    dpi: int,
    percentile: float,
    contrast_exponent: float,
) -> None:
    display_matrix = matrix.detach()
    if contrast_exponent != 1:
        display_matrix = display_matrix.sign() * display_matrix.abs().pow(contrast_exponent)

    values = display_matrix.cpu().numpy()
    abs_limit = float(torch.quantile(display_matrix.abs().flatten().cpu(), percentile / 100))
    if abs_limit == 0:
        abs_limit = None

    size = max(4, min(14, values.shape[0] / 128))
    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    ax.imshow(
        values,
        cmap="coolwarm",
        interpolation="nearest",
        vmin=-abs_limit if abs_limit else None,
        vmax=abs_limit,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate coolwarm visualizations for Laplacian and Steklov matrices."
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=DEFAULT_MESH,
        help=f"Mesh path (default: {DEFAULT_MESH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: scripts/<mesh-name>)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for matrix construction")
    parser.add_argument("--modes", type=int, default=K, help=f"Steklov eigenmodes (default: {K})")
    parser.add_argument(
        "--interior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use interior Steklov modes (default: false)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    parser.add_argument(
        "--percentile",
        type=float,
        default=100,
        help="Absolute value percentile for symmetric color scaling (default: 100)",
    )
    parser.add_argument(
        "--contrast-exponent",
        type=float,
        default=1,
        help=(
            "Apply sign(x) * abs(x) ** exponent before plotting; values below 1 boost "
            "small-magnitude contrast (default: 1)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.contrast_exponent <= 0:
        raise ValueError("--contrast-exponent must be positive")

    V, F = load_mesh(args.mesh, device=args.device)
    laplacian = cotan_laplacian_robust(V, F).to_dense()
    steklov = steklov_matrix(V, F, interior=args.interior, modes=args.modes)

    stem = args.mesh.stem
    output_dir = args.output_dir or SCRIPT_DIR / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    laplacian_path = output_dir / f"{stem}_laplacian.png"
    steklov_path = output_dir / f"{stem}_steklov.png"

    save_matrix_visualization(
        laplacian,
        laplacian_path,
        dpi=args.dpi,
        percentile=args.percentile,
        contrast_exponent=args.contrast_exponent,
    )
    save_matrix_visualization(
        steklov,
        steklov_path,
        dpi=args.dpi,
        percentile=args.percentile,
        contrast_exponent=args.contrast_exponent,
    )

    print(f"Wrote {laplacian_path}")
    print(f"Wrote {steklov_path}")


if __name__ == "__main__":
    main()

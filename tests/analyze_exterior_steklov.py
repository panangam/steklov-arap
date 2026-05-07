import argparse

import torch

from steklov_arap.arap import load_mesh
from steklov_arap.steklov import load_cached_steklov_dtn


def build_dtn_dense(cached, dtype, device):
    evals = cached["evals_int"].to(dtype=dtype, device=device).clone()
    evecs = cached["evecs_int"].to(dtype=dtype, device=device)
    mass = cached["mass"].to(dtype=dtype, device=device)

    evals[0] = 0  # first eigenvalue can be slightly negative numerically

    # Lambda = M Phi diag(lambda) Phi^T M  (matches steklov.py:244-246)
    dtn = (mass[:, None] * evecs * evals[None, :]) @ (evecs.mT * mass[None, :])
    return dtn, evals, mass


def summarize(name, M):
    n = M.shape[0]
    diag = torch.diagonal(M)
    off = M - torch.diag(diag)
    off_total = n * n - n
    pos_off = (off > 0).sum().item()
    neg_off = (off < 0).sum().item()
    zero_off = off_total - pos_off - neg_off
    row_sum = M.sum(dim=1)
    sym_err = (M - M.T).abs().max().item()

    print(f"--- {name} ---")
    print(f"  shape:              {tuple(M.shape)}")
    print(f"  dtype:              {M.dtype}")
    print(f"  symmetry err:       {sym_err:.3e}")
    print(f"  row sum max abs:    {row_sum.abs().max().item():.3e}")
    print(f"  row sum mean:       {row_sum.mean().item():.3e}")
    print(f"  row sum min/max:    {row_sum.min().item():.3e} / {row_sum.max().item():.3e}")
    print(f"  diag min/max/mean:  {diag.min().item():.3e} / {diag.max().item():.3e} / {diag.mean().item():.3e}")
    print(f"  off-diag min/max:   {off.min().item():.3e} / {off.max().item():.3e}")
    print(f"  off-diag mean abs:  {off.abs().mean().item():.3e}")
    print(f"  off-diag positive:  {pos_off}/{off_total} ({100*pos_off/off_total:.2f}%)")
    print(f"  off-diag negative:  {neg_off}/{off_total} ({100*neg_off/off_total:.2f}%)")
    print(f"  off-diag zero:      {zero_off}/{off_total} ({100*zero_off/off_total:.2f}%)")
    print(f"  frobenius norm:     {torch.linalg.norm(M).item():.3e}")

    # Diagonal dominance check (a Laplacian-like operator should have |L_ii| >= sum_j!=i |L_ij|)
    abs_off_row_sum = off.abs().sum(dim=1)
    dom_margin = diag.abs() - abs_off_row_sum
    print(f"  diag-dominance margin min/max: {dom_margin.min().item():.3e} / {dom_margin.max().item():.3e}")
    print(f"  rows that are diag-dominant:   {(dom_margin >= 0).sum().item()}/{n}")


def main():
    parser = argparse.ArgumentParser(description="Analyze the exterior Steklov DtN operator on a mesh.")
    parser.add_argument(
        "mesh",
        nargs="?",
        default="meshes/sphere.obj",
        help="Path to mesh file (default: meshes/sphere.obj)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", "-K", type=int, default=128, help="Number of Steklov modes (default: 128)")
    args = parser.parse_args()

    V, F = load_mesh(args.mesh, device=args.device)
    print(f"Loaded {args.mesh}: V={tuple(V.shape)} F={tuple(F.shape)} device={V.device}")
    print(f"Requested modes: K={args.modes}")

    cached = load_cached_steklov_dtn(V, F, interior=False, K=args.modes)
    dtn, evals, mass = build_dtn_dense(cached, dtype=V.dtype, device=V.device)

    print(f"Steklov eigenvalues: count={evals.numel()} "
          f"min={evals.min().item():.3e} max={evals.max().item():.3e}")
    print(f"Mass: min={mass.min().item():.3e} max={mass.max().item():.3e} "
          f"sum={mass.sum().item():.3e}")

    summarize("exterior Steklov DtN (Lambda = M Phi diag(lambda) Phi^T M)", dtn)


if __name__ == "__main__":
    main()

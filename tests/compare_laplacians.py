import argparse

import torch

from steklov_arap.arap import (
    load_mesh,
    cotan_laplacian,
    cotan_laplacian_abs,
    cotan_laplacian_robust,
)


def to_dense(L):
    return L.to_dense()


def summarize(name, L):
    Ld = to_dense(L)
    print(f"--- {name} ---")
    print(f"  shape:        {tuple(Ld.shape)}")
    print(f"  nnz:          {L._nnz()}")
    print(f"  dtype:        {Ld.dtype}")
    print(f"  row sum max:  {Ld.sum(dim=1).abs().max().item():.3e}")
    print(f"  symmetry err: {(Ld - Ld.T).abs().max().item():.3e}")
    diag = torch.diagonal(Ld)
    print(f"  diag min/max: {diag.min().item():.3e} / {diag.max().item():.3e}")
    off = Ld - torch.diag(diag)
    print(f"  off-diag min/max: {off.min().item():.3e} / {off.max().item():.3e}")


def compare(name_a, La, name_b, Lb):
    A = to_dense(La)
    B = to_dense(Lb)
    diff = (A - B).abs()
    denom = A.abs().max().clamp(min=1e-12)
    print(f"=== {name_a}  vs  {name_b} ===")
    print(f"  max abs diff:      {diff.max().item():.3e}")
    print(f"  mean abs diff:     {diff.mean().item():.3e}")
    print(f"  max rel diff:      {(diff.max() / denom).item():.3e}")
    print(f"  frobenius diff:    {torch.linalg.norm(A - B).item():.3e}")
    print(f"  frobenius A:       {torch.linalg.norm(A).item():.3e}")
    print(f"  frobenius B:       {torch.linalg.norm(B).item():.3e}")


def main():
    parser = argparse.ArgumentParser(description="Compare cotan Laplacian variants on a mesh.")
    parser.add_argument(
        "mesh",
        nargs="?",
        default="meshes/sphere.obj",
        help="Path to mesh file (default: meshes/sphere.obj)",
    )
    args = parser.parse_args()

    V, F = load_mesh(args.mesh)
    print(f"Loaded {args.mesh}: V={tuple(V.shape)} F={tuple(F.shape)}")

    L_none = cotan_laplacian(V, F, kind=None)
    L_abs = cotan_laplacian_abs(V, F)
    L_robust = cotan_laplacian_robust(V, F)

    summarize("cotan_laplacian(kind=None)", L_none)
    summarize("cotan_laplacian_abs", L_abs)
    summarize("cotan_laplacian_robust", L_robust)

    compare("kind=None", L_none, "abs", L_abs)
    compare("kind=None", L_none, "robust", L_robust)
    compare("abs", L_abs, "robust", L_robust)


if __name__ == "__main__":
    main()

import argparse

import torch

from steklov_arap.arap import load_mesh
from steklov_arap.steklov import load_cached_steklov_dtn


def to_dense(M):
    if M.is_sparse:
        return M.to_dense()
    return M


def summarize(name, M):
    Md = to_dense(M)
    n = Md.shape[0]
    diag = torch.diagonal(Md)
    off = Md - torch.diag(diag)
    off_total = n * n - n
    pos_off = (off > 0).sum().item()
    neg_off = (off < 0).sum().item()
    zero_off = off_total - pos_off - neg_off
    row_sum = Md.sum(dim=1)
    sym_err = (Md - Md.T).abs().max().item()
    abs_off_row_sum = off.abs().sum(dim=1)
    dom_margin = diag.abs() - abs_off_row_sum

    print(f"--- {name} ---")
    print(f"  shape:              {tuple(Md.shape)}")
    print(f"  dtype:              {Md.dtype}")
    print(f"  symmetry err:       {sym_err:.3e}")
    print(f"  row sum max abs:    {row_sum.abs().max().item():.3e}")
    print(f"  row sum min/max:    {row_sum.min().item():.3e} / {row_sum.max().item():.3e}")
    print(f"  diag min/max/mean:  {diag.min().item():.3e} / {diag.max().item():.3e} / {diag.mean().item():.3e}")
    print(f"  diag negative cnt:  {(diag < 0).sum().item()}/{n}")
    print(f"  off-diag min/max:   {off.min().item():.3e} / {off.max().item():.3e}")
    print(f"  off-diag mean abs:  {off.abs().mean().item():.3e}")
    print(f"  off-diag positive:  {pos_off}/{off_total} ({100*pos_off/off_total:.2f}%)")
    print(f"  off-diag negative:  {neg_off}/{off_total} ({100*neg_off/off_total:.2f}%)")
    print(f"  off-diag zero:      {zero_off}/{off_total} ({100*zero_off/off_total:.2f}%)")
    print(f"  frobenius norm:     {torch.linalg.norm(Md).item():.3e}")
    print(f"  diag-dominance margin min/max: {dom_margin.min().item():.3e} / {dom_margin.max().item():.3e}")
    print(f"  diag-dominant rows: {(dom_margin >= 0).sum().item()}/{n}")
    print(f"  strict dominant:    {(dom_margin > 0).sum().item()}/{n}")

    # M-matrix check: symmetric + non-positive off-diag + diag-dominant + positive diag.
    is_sym = sym_err < 1e-5 * Md.abs().max().item()
    nonpos_off = (off <= 1e-9).all().item()
    pos_diag = (diag > 0).all().item()
    diag_dom = (dom_margin >= -1e-9).all().item()
    print(f"  M-matrix conditions: sym={is_sym} nonpos_off={nonpos_off} "
          f"pos_diag={pos_diag} diag_dom={diag_dom}")

    # PSD check via smallest eigenvalue (only if dense and not too large).
    if Md.shape[0] <= 4000:
        Ms = 0.5 * (Md + Md.T)
        evals = torch.linalg.eigvalsh(Ms.double())
        print(f"  eigenvalues min/max:    {evals.min().item():.3e} / {evals.max().item():.3e}")
        print(f"  negative eigenvalues:   {(evals < -1e-8).sum().item()}/{n}")
        print(f"  near-zero eigenvalues:  {(evals.abs() < 1e-8).sum().item()}/{n}")


def analyze(mesh_path, K, interior, device):
    V, F = load_mesh(mesh_path, device=device)
    print(f"\n========== {mesh_path} (interior={interior}, K={K}) ==========")
    print(f"V={tuple(V.shape)} F={tuple(F.shape)}")

    cached = load_cached_steklov_dtn(V, F, interior=interior, K=K)

    S = cached["S_int"].to(dtype=V.dtype, device=V.device)
    mass = cached["mass"].to(dtype=V.dtype, device=V.device)
    evals = cached["evals_int"].to(dtype=V.dtype, device=V.device)

    print(f"S shape: {tuple(S.shape)} (sparse={S.is_sparse})")
    print(f"mass: min={mass.min().item():.3e} max={mass.max().item():.3e} "
          f"sum={mass.sum().item():.3e}")
    print(f"evals: count={evals.numel()} min={evals.min().item():.3e} "
          f"max={evals.max().item():.3e}")

    summarize("Stiffness matrix S_int", S)

    Sd = to_dense(S)
    if Sd.shape[0] == evals.numel():
        diag_evals = torch.diag(evals)
        diff = (Sd - diag_evals).abs()
        denom = Sd.abs().max().clamp(min=1e-30)
        print(f"  ||S - diag(evals)||_max:    {diff.max().item():.3e}")
        print(f"  ||S - diag(evals)||_max / ||S||_max: {(diff.max() / denom).item():.3e}")
        print(f"  ||S - diag(evals)||_F / ||S||_F:     "
              f"{(torch.linalg.norm(Sd - diag_evals) / torch.linalg.norm(Sd).clamp(min=1e-30)).item():.3e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Steklov stiffness matrix S properties.")
    parser.add_argument("--modes", "-K", type=int, default=128)
    parser.add_argument("--interior", action="store_true", help="Use interior basis (default: exterior)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--meshes", nargs="+", default=["meshes/sphere.obj", "meshes/bunny.obj"])
    args = parser.parse_args()

    for mesh in args.meshes:
        analyze(mesh, K=args.modes, interior=args.interior, device=args.device)


if __name__ == "__main__":
    main()

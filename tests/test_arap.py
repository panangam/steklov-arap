import torch

import pytest

from steklov_arap import (
    cotan_laplacian_robust,
    rots_from_verts,
    verts_from_rots,
    remove_rows_cols,
    create_constrained_solver,
    SolverHelper,
)


def test_rots_from_verts_identity(mesh):
    """When V and Vp are identical, every per-vertex rotation should be identity."""
    V, F, L = mesh

    R = rots_from_verts(V, V, L)

    I = (
        torch.eye(3, dtype=V.dtype, device=V.device)
        .unsqueeze(0)
        .expand(V.shape[0], -1, -1)
    )
    assert torch.allclose(
        R, I, atol=1e-5
    ), f"Expected all identity rotations, max deviation: {(R - I).abs().max().item()}"


def test_verts_from_rots_identity(mesh):
    """With identity rotations and handles fixed at original positions, vertices should be unchanged."""
    V, F, L = mesh
    radius = torch.norm(V - V.mean(dim=0, keepdim=True), dim=1).max().item()

    V_clone = V.clone()

    R = (
        torch.eye(3, dtype=V.dtype, device=V.device)
        .unsqueeze(0)
        .expand(V.shape[0], -1, -1)
        .contiguous()
    )
    handle_idxs = [2]
    V_handle = V_clone[handle_idxs]
    solver = create_constrained_solver(L, handle_idxs)

    V_new = verts_from_rots(V_clone, R, L, handle_idxs, V_handle, solver)

    assert torch.allclose(
        V_clone, V, atol=1e-4
    ), f"Expected input vertices to not change: {(V_clone - V).abs().max().item() / radius}"

    assert torch.allclose(
        V_new, V, atol=1e-4
    ), f"Expected unchanged vertices, max rel deviation: {(V_new - V).abs().max().item() / radius}"


def test_remove_rows_cols(mesh):
    """Removed rows/cols should be absent and remaining entries should match the original dense matrix."""
    V, F, L = mesh

    remove_idx = [0, 5, 100]
    L_reduced = remove_rows_cols(L, remove_idx)

    L_dense = L.to_dense()
    keep = [i for i in range(L.shape[0]) if i not in set(remove_idx)]
    L_expected = L_dense[keep][:, keep]

    assert L_reduced.shape == (
        len(keep),
        len(keep),
    ), f"Expected shape {(len(keep), len(keep))}, got {L_reduced.shape}"
    assert torch.allclose(L_reduced.to_dense(), L_expected, atol=1e-6), (
        f"Reduced matrix entries don't match dense reference, max deviation: "
        f"{(L_reduced.to_dense() - L_expected).abs().max().item()}"
    )


def test_solver_helper_sparse_cpu_fallback():
    """The fallback sparse solver should work on CPU without torch.sparse.spsolve."""
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([4.0, 1.0, 1.0, 3.0])
    A = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = torch.empty_like(b)

    SolverHelper(A).solve(b, x)

    assert torch.allclose(A.to_dense() @ x, b, atol=1e-6)


def test_cotan_laplacian_robust_returns_sparse_coo(mesh):
    """The robust Laplacian should be returned as a coalesced PyTorch COO tensor."""
    V, F, _ = mesh
    L = cotan_laplacian_robust(V, F)

    assert L.layout == torch.sparse_coo
    assert L.is_coalesced()
    assert L.shape == (V.shape[0], V.shape[0])
    assert L.dtype == V.dtype
    assert L.device == V.device
    assert L._nnz() > 0

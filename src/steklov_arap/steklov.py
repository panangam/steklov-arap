from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import torch

from steklovnet.geometry.mesh import Mesh

from .arap import cotan_laplacian_robust, cotan_laplacian_abs

K = 386
MAX_POINTS = 1 << 14
BLUE_NOISE_RADIUS = 0.01
TOTAL_SAMPLES = 10_000_000
CHUNK_SAMPLES = 1_000_000
TOL = 1e-6
MAX_ITERS = 1024
ANGULAR_CLAMP = 1e-3
BISECTION_ITERS = 10
KNN_K = 8
KNN_SIGMA = 0.025
SEED = 6336
THREADS = 128
GWN_EPS = 1e-5


class DenseCholeskySolver:
    def __init__(self, A):
        self.L = torch.linalg.cholesky(A)

    def solve(self, b, x_out):
        if b.ndim == 1:
            x_out[:] = torch.cholesky_solve(b[:, None], self.L).squeeze(-1)
            return
        x_out[:] = torch.cholesky_solve(b, self.L)


def absify_diagonal_(M):
    M.abs_().negative_()
    M.fill_diagonal_(0)
    M.diagonal().copy_(-M.sum(dim=1))
    return M


def load_cached_steklov_dtn(V, F, interior=True, K=K):
    V_cpu = V.detach().contiguous().cpu()
    F_cpu = F.detach().contiguous().cpu()

    hasher = hashlib.sha256()
    for tensor in (V_cpu, F_cpu):
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
        hasher.update(interior.to_bytes(1))
    hasher.update(f"K={K}".encode("utf-8"))

    cache_path = Path(tempfile.gettempdir()) / f"steklov_dtn_{hasher.hexdigest()}.pt"
    expected_keys = {"evals_int", "evecs_int", "S_int", "mass"}

    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu")
            if expected_keys.issubset(cached):
                return cached
        except Exception:
            pass

    mesh = Mesh.load(V, F, str(V.device))
    mode_count = min(K, V.shape[0] - 1)
    evals_int, evecs_int, S_int, basis_int, _ = mesh.steklov_eigenmodes_mesh(
        interior=interior,
        galerkin_basis='robust_laplacian',
        K=mode_count,
        angular_clamp=ANGULAR_CLAMP,
        total_samples=TOTAL_SAMPLES,
        tol=TOL,
        max_iters=MAX_ITERS,
        seed=SEED,
        threads=THREADS,
        bisection_iters=BISECTION_ITERS,
        chunk_samples=CHUNK_SAMPLES,
    )

    cached = {
        "evals_int": evals_int.detach().cpu(),
        "evecs_int": evecs_int.detach().cpu(),
        "S_int": S_int.detach().cpu(),
        "mass": basis_int.mass.detach().cpu(),
    }
    torch.save(cached, cache_path)
    return cached


def load_cached_dtn_operator_dense(V, F, interior=True):
    device = V.device
    dtype = V.dtype
    
    cached_dtn = load_cached_steklov_dtn(V, F, interior)

    steklov_evals = cached_dtn["evals_int"].to(dtype=dtype, device=device)
    steklov_evecs = cached_dtn["evecs_int"].to(dtype=dtype, device=device)
    mass = cached_dtn["mass"].to(dtype=dtype, device=device)

    mass = torch.pow(mass, 2/3)

    steklov_evals[0] = 0  # for some reason I got negative first eigenvalue sometimes :(

    dtn = (mass[:, None] * steklov_evecs * steklov_evals[None, :]) @ (
        steklov_evecs.mT * mass[None, :]
    )

    return dtn


def rots_from_verts_dense(V, Vp, L, eps=1e-7):
    # Dense analogue of rots_from_verts() for a dense Laplacian-like operator.
    device = V.device
    dtype = V.dtype

    weights = torch.clamp(-L, min=eps)
    E = V[:, None, :] - V[None, :, :]
    Ep = Vp[:, None, :] - Vp[None, :, :]
    S = (weights[:, :, None, None] * E[:, :, :, None] * Ep[:, :, None, :]).sum(dim=1)

    U, Sigma, Vh = torch.linalg.svd(S)

    R_test = Vh.mT @ U.mT
    sign = torch.sign(torch.det(R_test))
    U_mod = U.clone()
    U_mod[:, :, 2] *= sign[:, None]
    R = Vh.mT @ U_mod.mT

    degen = (S.norm(dim=(1, 2)) < 1e-4) | (Sigma[:, 2] < 1e-4)
    R[degen] = torch.eye(3, dtype=dtype, device=device)

    return R


def verts_from_rots_dense(
    V,
    R,
    L,
    L_row_sum,
    handle_idxs,
    V_handle,
    handle_coupling,
    solver,
):
    device = V.device
    dtype = V.dtype

    # rotated_rest = (R @ V[:, :, None]).squeeze(-1)
    # LV_rest = L @ V
    # weighted_rotations = torch.einsum('ij,jab->iab', L, R)
    # b = 0.5 * (
    #     L_row_sum[:, None] * rotated_rest
    #     + (weighted_rotations @ V[:, :, None]).squeeze(-1)
    #     - (R @ LV_rest[:, :, None]).squeeze(-1)
    #     - L @ rotated_rest
    # )

    include_mask = torch.ones(V.shape[0], dtype=torch.bool, device=device)
    include_mask[handle_idxs] = False

    if not include_mask.any():
        V_new = V.clone()
        V_new[handle_idxs] = V_handle
        return V_new

    R_sum = R[:, None, :, :] + R[None, :, :, :]
    p_diff = V[:, None, :] - V[None, :, :]
    b = torch.einsum('ijyx,ijx->ijy', -L[..., None, None]/2 * R_sum, p_diff).sum(axis=1)

    if handle_idxs:
        b_free = b[include_mask] - handle_coupling @ V_handle
    else:
        b_free = b

    V_solved = torch.empty_like(b_free)
    solver.solve(b_free, V_solved)

    V_new = torch.empty_like(V, dtype=dtype, device=device)
    V_new[include_mask] = V_solved
    V_new[handle_idxs] = V_handle
    return V_new


class ARAPManagerSteklov:
    """
    Stateful manager for Steklov-DtN ARAP deformation, mirroring ARAPManager.
    """

    def __init__(
        self,
        V_rest,
        F,
        device="cuda",
        float_dtype=torch.float32,
        kind='robust',
        interior=False,
        alpha=0.1,
    ):
        del kind

        self.device = device
        self.V_rest = torch.as_tensor(V_rest, dtype=float_dtype, device=device)
        self.F = torch.as_tensor(F, dtype=torch.int32, device=device)
        self.handle_idxs = []
        self.V_handle = torch.empty((0, 3), dtype=self.V_rest.dtype, device=self.device)
        self.V_deformed = self.V_rest.clone()

        self.mesh = None
        self.steklov_evals = None
        self.steklov_evecs = None
        self.steklov_stiffness = None
        self.mass = None
        self.L = None
        self.L_row_sum = None
        self.system_matrix = None
        self.handle_coupling = torch.empty(
            (self.V_rest.shape[0], 0), dtype=self.V_rest.dtype, device=self.device
        )
        self.solver = None
        self.alpha = alpha
        self.interior = interior

        self._update_operator()
        self._update_system_matrix()

    def _update_operator(self):
        cached_dtn = load_cached_steklov_dtn(self.V_rest, self.F, interior=self.interior)

        self.steklov_evals = cached_dtn["evals_int"].to(dtype=self.V_rest.dtype, device=self.device)
        self.steklov_evecs = cached_dtn["evecs_int"].to(dtype=self.V_rest.dtype, device=self.device)
        self.steklov_stiffness = cached_dtn["S_int"].to(dtype=self.V_rest.dtype, device=self.device)
        self.mass = cached_dtn["mass"].to(dtype=self.V_rest.dtype, device=self.device)

        self.steklov_evals[0] = 0  # for some reason I got negative first eigenvalue sometimes :(

        # # Dense DtN operator Lambda = Phi diag(lambda) Phi^T M.
        # self.L = (self.mass[:, None] * self.steklov_evecs * self.steklov_evals[None, :]) @ (
        #     self.steklov_evecs.mT * self.mass[None, :]
        # )

        # # make sure we don't have negative weights
        # self.L.abs_().negative_()
        # self.L.fill_diagonal_(0)
        # self.L.diagonal().copy_(-self.L.sum(dim=0))

        bisteklov = (self.mass[:, None] * self.steklov_evecs * torch.pow(self.steklov_evals[None, :], 2)) @ (
            self.steklov_evecs.mT * self.mass[None, :]
        )
        # turn into M-matrix
        # absify_diagonal_(dtn)

        laplacian = cotan_laplacian_robust(self.V_rest, self.F)

        self.L = (1-self.alpha)*laplacian.to_dense() + self.alpha*bisteklov
        # import ipdb; ipdb.set_trace()

        self.L_row_sum = self.L.sum(dim=1)

    def _update_system_matrix(self):
        n_verts = self.V_rest.shape[0]
        include_mask = torch.ones(n_verts, dtype=torch.bool, device=self.device)
        include_mask[self.handle_idxs] = False

        if not include_mask.any():
            self.system_matrix = torch.empty((0, 0), dtype=self.V_rest.dtype, device=self.device)
            self.handle_coupling = torch.empty(
                (0, len(self.handle_idxs)), dtype=self.V_rest.dtype, device=self.device
            )
            self.solver = None
            return

        if self.handle_idxs:
            self.system_matrix = self.L[include_mask][:, include_mask]
            self.handle_coupling = self.L[include_mask][:, self.handle_idxs]
        else:
            self.system_matrix = self.L + TOL * torch.eye(
                n_verts, dtype=self.V_rest.dtype, device=self.device
            )
            self.handle_coupling = torch.empty(
                (n_verts, 0), dtype=self.V_rest.dtype, device=self.device
            )

        try:
            self.solver = DenseCholeskySolver(self.system_matrix)
        except RuntimeError:
            self.system_matrix = self.system_matrix + TOL * torch.eye(
                self.system_matrix.shape[0],
                dtype=self.V_rest.dtype,
                device=self.device,
            )
            self.solver = DenseCholeskySolver(self.system_matrix)

    def set_handle_constraints(self, handle_idxs, V_handle):
        if isinstance(handle_idxs, torch.Tensor):
            handle_idxs = handle_idxs.detach().cpu().tolist()
        else:
            handle_idxs = [int(idx) for idx in handle_idxs]

        if handle_idxs != self.handle_idxs:
            self.handle_idxs = handle_idxs
            self._update_system_matrix()
        self.set_handle_positions(V_handle)

    def set_handle_positions(self, V_handle):
        self.V_handle = torch.as_tensor(
            V_handle, dtype=self.V_rest.dtype, device=self.device
        )
        self.V_deformed[self.handle_idxs] = self.V_handle

    def set_rest_state(self):
        self.V_rest = self.V_deformed.clone()
        self._update_operator()
        self._update_system_matrix()

    def reset_to_rest_state(self):
        self.V_deformed = self.V_rest.clone()
        self.V_handle = self.V_rest[self.handle_idxs].clone()
        self.V_deformed[self.handle_idxs] = self.V_handle
        return self.V_deformed

    def iterate(self):
        R = rots_from_verts_dense(self.V_rest, self.V_deformed, self.L)
        self.V_deformed = verts_from_rots_dense(
            self.V_rest,
            R,
            self.L,
            self.L_row_sum,
            self.handle_idxs,
            self.V_handle,
            self.handle_coupling,
            self.solver,
        )


__all__ = ["ARAPManagerSteklov"]

import torch
import igl
import cholespy
import robust_laplacian
import scipy.sparse
import scipy.sparse.linalg


def load_mesh(path, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    V, F = igl.read_triangle_mesh(path)
    V = torch.tensor(V, dtype=torch.float32, device=device)
    F = torch.tensor(F, dtype=torch.int32, device=device)
    return V, F


def cotan_laplacian(V, F, kind=None):
    device = V.device
    n_verts = V.shape[0]

    e_01 = V[F[:, 1]] - V[F[:, 0]]
    e_12 = V[F[:, 2]] - V[F[:, 1]]
    e_20 = V[F[:, 0]] - V[F[:, 2]]

    cot_0 = (
        (e_01 * (-e_20)).sum(dim=-1) / torch.norm(e_01.cross(-e_20, dim=-1), dim=-1) / 2
    )
    cot_1 = (
        (e_12 * (-e_01)).sum(dim=-1) / torch.norm(e_12.cross(-e_01, dim=-1), dim=-1) / 2
    )
    cot_2 = (
        (e_20 * (-e_12)).sum(dim=-1) / torch.norm(e_20.cross(-e_12, dim=-1), dim=-1) / 2
    )

    if kind == 'abs':
        cot_0.abs_()
        cot_1.abs_()
        cot_2.abs_()

    L_indices = torch.concat(
        [
            F[:, [1, 1]],
            F[:, [2, 2]],
            F[:, [1, 2]],
            F[:, [2, 1]],
            F[:, [0, 0]],
            F[:, [2, 2]],
            F[:, [0, 2]],
            F[:, [2, 0]],
            F[:, [0, 0]],
            F[:, [1, 1]],
            F[:, [0, 1]],
            F[:, [1, 0]],
        ],
        axis=0,
    ).T

    L_values = torch.concat(
        [
            cot_0,
            cot_0,
            -cot_0,
            -cot_0,
            cot_1,
            cot_1,
            -cot_1,
            -cot_1,
            cot_2,
            cot_2,
            -cot_2,
            -cot_2,
        ],
        axis=0,
    )

    L = torch.sparse_coo_tensor(
        L_indices, L_values, size=(n_verts, n_verts), device=device
    ).coalesce()

    return L


def cotan_laplacian_abs(V, F):
    return cotan_laplacian(V, F, 'abs')


def cotan_laplacian_robust(V, F):
    V_np = V.detach().cpu().numpy()
    F_np = F.detach().cpu().numpy()
    L_sp, _ = robust_laplacian.mesh_laplacian(V_np, F_np)
    L_coo = L_sp.tocoo()
    indices = torch.stack(
        [
            torch.as_tensor(L_coo.row, dtype=torch.long, device=V.device),
            torch.as_tensor(L_coo.col, dtype=torch.long, device=V.device),
        ]
    )
    values = torch.as_tensor(L_coo.data, dtype=V.dtype, device=V.device)
    return torch.sparse_coo_tensor(
        indices, values, size=L_coo.shape, device=V.device
    ).coalesce()


def test_cotan_laplacian():
    V, F = load_mesh("meshes/sphere.obj")
    L = cotan_laplacian(V, F)
    assert torch.isclose(L.sum(dim=0).to_dense(), torch.tensor(0.0), atol=1e-6).all()
    assert torch.isclose(L.sum(dim=1).to_dense(), torch.tensor(0.0), atol=1e-6).all()


def rots_from_verts(V, Vp, L, eps=1e-7):
    # Compute per-vertex optimal rotation matrices via SVD (ARAP local step).
    device = V.device
    dtype = V.dtype

    L_idxs = L.indices()
    L_vals = L.values()
    
    # clamp weights to be non-negative for stability
    L_vals = torch.clamp(-L_vals, min=eps)  # ensure positive weights for stability

    # Edge vectors: e_ij = p_i - p_j, shape [nnz, 3]
    E = V[L_idxs[0]] - V[L_idxs[1]]
    Ep = Vp[L_idxs[0]] - Vp[L_idxs[1]]

    # Weighted outer products w_ij * e_ij * e'_ij^T, shape [nnz, 3, 3]
    E_EpT = E[:, :, None] * (L_vals[:, None, None]) * Ep[:, None, :]

    # Covariance matrix S_i = sum_j w_ij * e_ij * e'_ij^T, accumulated per vertex i
    S = torch.zeros((V.shape[0], 3, 3), dtype=dtype, device=device)
    S.index_add_(0, L_idxs[0], E_EpT)
    # S += torch.eye(3, dtype=dtype, device=device)[None] * eps  # regularization for stability

    # Optimal rotation R_i = V_i @ U_i^T minimises sum_j w_ij ||R e_ij - e'_ij||^2
    U, Sigma, Vh = torch.linalg.svd(S)

    # Ensure proper rotation (det = +1), not reflection (det = -1):
    # flip the sign of the last column of U when the candidate rotation reflects.
    # D = torch.ones(V.shape[0], 3, dtype=dtype, device=device)
    # D[:, 2] = torch.det(Vh.mT @ U.mT)
    # R = Vh.mT @ (D[:, None, :] * U).mT
    R_test = Vh.mT @ U.mT
    sign = torch.sign(torch.det(R_test))
    U_mod = U.clone()
    U_mod[:, :, 2] *= sign[:, None]
    R = Vh.mT @ U_mod.mT

    degen = (S.norm(dim=(1, 2)) < 1e-4) | (Sigma[:, 2] < 1e-4)
    R[degen] = torch.eye(3, dtype=dtype, device=device)

    return R


def create_constrained_solver(L, handle_idxs):
    L_constrained = remove_rows_cols(L, handle_idxs)
    try:
        solver_class = (
            cholespy.CholeskySolverF
            if L_constrained.dtype == torch.float32
            else cholespy.CholeskySolverD
        )
        solver = solver_class(
            L_constrained.shape[0],
            L_constrained.indices()[0],
            L_constrained.indices()[1],
            L_constrained.values(),
            cholespy.MatrixType.COO,
        )
    except ValueError as e:
        try:
            n = L_constrained.shape[0]
            eye_idx = torch.arange(n, device=L_constrained.device)
            reg = torch.sparse_coo_tensor(
                torch.stack([eye_idx, eye_idx]),
                torch.full((n,), 1e-6, dtype=L_constrained.dtype, device=L_constrained.device),
                L_constrained.shape,
            )
            L_constrained = (L_constrained + reg).coalesce()
            print("Warning: Cholesky factorization failed, added diagonal regularization and retrying.")
            solver = solver_class(
                L_constrained.shape[0],
                L_constrained.indices()[0],
                L_constrained.indices()[1],
                L_constrained.values(),
                cholespy.MatrixType.COO,
            )
        except ValueError as e:
            print("Warning: Cholesky factorization failed, using sparse solver fallback.")
            print(e)
            solver = SolverHelper(L_constrained)
    return solver


def verts_from_rots(V, R, L, handle_idxs, V_handle, solver):
    device = V.device
    dtype = V.dtype

    L_idxs = L.indices()
    L_vals = L.values()

    R_sum = R[L_idxs[0]] + R[L_idxs[1]]
    E = V[L_idxs[0]] - V[L_idxs[1]]
    w_R_p = -L_vals[:, None] / 2 * (R_sum @ E[..., None]).squeeze()
    b = torch.zeros((V.shape[0], 3), dtype=dtype, device=device)
    b.index_add_(0, L_idxs[0], w_R_p)

    # # add constraints
    b_update = get_b_update_from_constraints(V_handle, L, handle_idxs)
    b_constrained = b - b_update
    include_mask = torch.ones(V.shape[0], dtype=torch.bool, device=device)
    include_mask[handle_idxs] = False
    b_constrained = b_constrained[include_mask]

    V_solved = torch.empty_like(b_constrained)
    solver.solve(b_constrained, V_solved)

    V_new = V.clone()
    V_new[include_mask] = V_solved
    V_new[handle_idxs] = V_handle

    return V_new


def get_b_update_from_constraints(V_handle, L, handle_idxs):
    # V_handle_idxs = torch.stack(
    #     torch.meshgrid(
    #         torch.tensor(handle_idxs, device=V_handle.device),
    #         torch.arange(V_handle.shape[1], device=V_handle.device),
    #     ),
    #     dim=0,
    # ).reshape(2, -1)
    # V_handle = torch.sparse_coo_tensor(
    #     V_handle_idxs, V_handle.flatten(), size=(L.shape[0], V_handle.shape[1]), device=V_handle.device
    # ).coalesce()

    V_handle_dense = torch.zeros(
        (L.shape[0], V_handle.shape[1]), dtype=V_handle.dtype, device=V_handle.device
    )
    V_handle_dense[handle_idxs] = V_handle
    b_update = L @ V_handle_dense

    return b_update


def remove_rows_cols(A: torch.Tensor, remove_idx: list[int]) -> torch.Tensor:
    idx = A.indices()  # (2, nnz)
    val = A.values()
    n = A.size(0)  # assumes square

    remove = torch.tensor(remove_idx, dtype=torch.int32, device=A.device)

    # Keep entries where neither row nor col is in remove_idx
    keep = ~torch.isin(idx[0], remove) & ~torch.isin(idx[1], remove)
    rows, cols = idx[0][keep], idx[1][keep]

    # Build old -> new index mapping
    keep_nodes = torch.tensor(
        [i for i in range(n) if i not in set(remove_idx)], device=A.device
    )
    old_to_new = torch.full((n,), -1, dtype=torch.long, device=A.device)
    old_to_new[keep_nodes] = torch.arange(len(keep_nodes), device=A.device)

    new_idx = torch.stack([old_to_new[rows], old_to_new[cols]])
    new_n = n - len(remove_idx)

    return torch.sparse_coo_tensor(
        new_idx, val[keep], (new_n, new_n), device=A.device
    ).coalesce()


class SolverHelper:
    def __init__(self, A):
        A_cpu = A.detach().cpu().to_sparse_coo().coalesce()
        indices = A_cpu.indices().numpy()
        values = A_cpu.values().numpy()
        A_scipy = scipy.sparse.coo_matrix(
            (values, (indices[0], indices[1])), shape=A_cpu.shape
        ).tocsc()
        self.solve_A = scipy.sparse.linalg.splu(A_scipy).solve

    def solve(self, b, x_out):
        b_cpu = b.detach().cpu()
        x = self.solve_A(b_cpu.numpy())
        x = torch.as_tensor(x, dtype=b_cpu.dtype)
        x_out[:] = x.to(device=x_out.device, dtype=x_out.dtype)


class ARAPManager:
    """
    stateful manager for ARAP deformation,
    holding the rest pose, handle constraints,
    and pre-factorised solver.
    """

    def __init__(self, V_rest, F, device="cuda", float_dtype=torch.float32, kind='robust'):
        self.device = device
        self.V_rest = torch.as_tensor(V_rest, dtype=float_dtype, device=device)
        self.F = torch.as_tensor(F, dtype=torch.int32, device=device)
        self.L_func = dict(
            robust=cotan_laplacian_robust,
            abs=cotan_laplacian_abs,
        )[kind]
        self.L = self.L_func(self.V_rest, self.F)
        self.handle_idxs = []
        self.V_handle = torch.empty((0, 3), dtype=self.V_rest.dtype, device=self.device)
        self.V_deformed = self.V_rest.clone()

        eps = 1e-4
        nV = self.V_rest.shape[0]
        sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
        L_eps = (self.L + sparse_eps_diag).coalesce()

        self.solver = create_constrained_solver(L_eps, self.handle_idxs)

    def set_handle_constraints(self, handle_idxs, V_handle):
        if handle_idxs != self.handle_idxs and len(handle_idxs) > 0:
            self.handle_idxs = handle_idxs
            self.solver = create_constrained_solver(self.L, handle_idxs)
        self.set_handle_positions(V_handle)

    def set_handle_positions(self, V_handle):
        self.V_handle = torch.as_tensor(
            V_handle, dtype=self.V_rest.dtype, device=self.device
        )
        self.V_deformed[self.handle_idxs] = self.V_handle

    def set_rest_state(self):
        self.V_rest = self.V_deformed.clone()
        self.L = self.L_func(self.V_rest, self.F)
        self.solver = create_constrained_solver(self.L, self.handle_idxs)

    def reset_to_rest_state(self):
        self.V_deformed = self.V_rest.clone()
        self.V_handle = self.V_rest[self.handle_idxs].clone()
        self.V_deformed[self.handle_idxs] = self.V_handle
        return self.V_deformed

    def iterate(self):
        R = rots_from_verts(self.V_rest, self.V_deformed, self.L)
        self.V_deformed = verts_from_rots(
            self.V_rest, R, self.L, self.handle_idxs, self.V_handle, self.solver
        )

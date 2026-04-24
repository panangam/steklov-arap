# %%
import matplotlib.pyplot as plt
import torch
import igl
import k3d
import cholespy
import scipy

from steklov_arap.arap import *

#!%load_ext autoreload  
#!%autoreload 3  

# %%
def load_mesh(path, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    V, F = igl.read_triangle_mesh(path)
    V = torch.tensor(V, dtype=torch.float32, device=device)
    F = torch.tensor(F, dtype=torch.int32, device=device)
    return V, F

# %%
V, F = load_mesh("meshes/armadillo.obj", "cuda")
L = cotan_laplacian_robust(V, F)

# %%
F_np = F.cpu().numpy()
V_iter = V.clone()
handle_idxs = [210, 181]
V_handle = V[handle_idxs].clone()
V_handle[0] += torch.tensor([0.0, 0.0, -1], device=V.device)
V_iter[handle_idxs] = V_handle 
solver = create_constrained_solver(L, handle_idxs)

for i in range(10):
    R = rots_from_verts(V, V_iter, L)
    V_iter = verts_from_rots(V, R, L, handle_idxs, V_handle, solver)

plot = k3d.plot()
plot += k3d.mesh(V.cpu().numpy(), F_np, color=0xff, opacity=0.5)
plot += k3d.mesh(V_iter.cpu().numpy(), F_np, color=0x00ff00, opacity=0.5)
# plot += k3d.points(V[handle_idxs].cpu().numpy(), point_size=0.1, color=0xffff00)
# plot += k3d.points(V_end[handle_idxs].cpu().numpy(), point_size=0.1, color=0xffff00)
plot += k3d.points(V_handle.cpu().numpy(), point_size=0.1, color=0xffff00)
plot

# %%
arap = ARAPManager(V, F, device="cuda")
arap.set_handle_constraints(handle_idxs, V_handle)
for _ in range(10):
    arap.iterate()
plot = k3d.plot()
plot += k3d.mesh(V.cpu().numpy(), F_np, color=0xff, opacity=0.5)
plot += k3d.mesh(arap.V_deformed.cpu().numpy(), F_np, color=0x00ff00, opacity=0.5)
# plot += k3d.points(V[handle_idxs].cpu().numpy(), point_size=0.1, color=0xffff00)
# plot += k3d.points(V_end[handle_idxs].cpu().numpy(), point_size=0.1, color=0xffff00)
plot += k3d.points(V_handle.cpu().numpy(), point_size=0.1, color=0xffff00)
plot


# %%
R_test = rots_from_verts(V, V, L)
error = torch.amax((R_test-torch.eye(3)).abs(), dim=(-1, -2))
plt.plot(error)

# %%
plot = k3d.plot()
plot += k3d.mesh(V.cpu().numpy(), F_np, color=0x00ff00, opacity=0.5)
plot += k3d.points([V[90]], point_size=0.1, color=0xffff00)
plot

# %%
# test laplacian eigenspectrum
L_scipy = scipy.sparse.coo_matrix((L.values().cpu().numpy(), (L.indices()[0].cpu().numpy(), L.indices()[1].cpu().numpy())), shape=L.shape)
M = igl.massmatrix(V.cpu().numpy(), F.cpu().numpy(), igl.MASSMATRIX_TYPE_DEFAULT)
evals, evecs = scipy.sparse.linalg.eigsh(L_scipy, M=M, k=100, which='SM')
plt.plot(evals)
plt.show()
k3d.mesh(V.cpu().numpy(), F.cpu().numpy(), attribute=evecs[:, 4].real)

# %% md
# ## Test laplacian with heat diffusion

# %%
def sparse_eye(n, device=None, dtype=torch.float32):
    idx = torch.arange(n, device=device)
    indices = torch.stack([idx, idx])   # shape [2, n]
    values = torch.ones(n, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

f = torch.zeros(V.shape[0], dtype=V.dtype, device=V.device)
f[210] = 1
I = sparse_eye(V.shape[0], device=V.device, dtype=V.dtype)
f2 = (I - 0.2 * L) @ (I - 0.2 * L) @ (I - 0.2 * L) @ f
k3d.mesh(V.cpu().numpy(), F.cpu().numpy(), attribute=f2.cpu().numpy())

# %%
solver = cholespy.CholeskySolverF(
    L.shape[0], L.indices()[0], L.indices()[1], L.values(), cholespy.MatrixType.COO
)
b = torch.ones(V.shape[0], dtype=V.dtype, device=V.device).reshape(-1, 1)
x = torch.ones(V.shape[0], dtype=V.dtype, device=V.device).reshape(-1, 1)
solver.solve(b, x)
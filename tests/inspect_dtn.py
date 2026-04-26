import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt

    from steklov_arap.steklov import load_cached_dtn_operator_dense
    from steklov_arap.arap import load_mesh, cotan_laplacian_robust

    return (
        cotan_laplacian_robust,
        load_cached_dtn_operator_dense,
        load_mesh,
        plt,
        torch,
    )


@app.cell
def _(cotan_laplacian_robust, load_cached_dtn_operator_dense, load_mesh):
    V, F = load_mesh('meshes/.obj', device='cuda')
    dtn_ext = load_cached_dtn_operator_dense(V, F, interior=False)
    dtn_int = load_cached_dtn_operator_dense(V, F, interior=True)
    L = cotan_laplacian_robust(V, F).to_dense()
    return L, V, dtn_int


@app.cell
def _(torch):
    def softened_signed_power(x: torch.Tensor, gamma: float = 1/3, eps: float = 1e-8):
        y = torch.sign(x) * ((torch.abs(x) + eps).pow(gamma) - eps**gamma)
        return y / y.abs().amax().clamp_min(1e-12)

    return (softened_signed_power,)


@app.cell
def _(L, V, dtn_int, plt, softened_signed_power, torch):
    off_diagonal_mask = ~torch.eye(V.shape[0], dtype=bool, device='cuda')
    alpha = 1
    blended = (1-alpha)*L + alpha*dtn_int

    plt.title('off-diagonal entries (blown up for visualization)')
    plt.hist(softened_signed_power(blended[off_diagonal_mask].cpu(), 1/10), bins=100)
    plt.show()
    return blended, off_diagonal_mask


@app.cell
def _(blended, off_diagonal_mask, torch):
    torch.pow(blended[off_diagonal_mask].cpu(), 1/3)
    return


if __name__ == "__main__":
    app.run()

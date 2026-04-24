import torch
import os

import pytest

from steklov_arap import load_mesh, cotan_laplacian


@pytest.fixture(
    params=["meshes/sphere.obj", "meshes/armadillo.obj", "meshes/teapot.obj"],
    ids=lambda p: os.path.basename(p).split(".")[0],
)
def mesh_path(request):
    return request.param


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return request.param


@pytest.fixture
def mesh(mesh_path, device):
    V, F = load_mesh(mesh_path, device=device)
    L = cotan_laplacian(V, F)
    return V, F, L

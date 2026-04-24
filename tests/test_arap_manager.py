import torch

from steklov_arap import ARAPManager


def test_init_deformed_equals_rest(mesh):
    """V_deformed should equal V_rest after construction."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    assert torch.allclose(manager.V_deformed, manager.V_rest)


def test_init_no_handles(mesh):
    """handle_idxs should be empty after construction."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    assert manager.handle_idxs == []


def test_set_handle_positions_updates_deformed(mesh):
    """set_handle_positions should immediately update V_deformed at handle indices."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    handle_idxs = [0, 1, 2]
    manager.set_handle_constraints(handle_idxs, V[handle_idxs])

    new_positions = V[handle_idxs] + 0.1
    manager.set_handle_positions(new_positions)

    assert torch.allclose(manager.V_deformed[handle_idxs], new_positions)


def test_iterate_identity_handles_unchanged(mesh):
    """With handles fixed at rest positions, iterating should leave V_deformed close to V_rest."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    handle_idxs = [0, 1, 2]
    manager.set_handle_constraints(handle_idxs, V[handle_idxs])

    for _ in range(3):
        manager.iterate()

    assert torch.allclose(manager.V_deformed, manager.V_rest, atol=1e-4), (
        f"Max deviation after iteration: {(manager.V_deformed - manager.V_rest).abs().max().item()}"
    )


def test_set_rest_state(mesh):
    """After set_rest_state, V_rest should equal the previous V_deformed."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    handle_idxs = [0]
    displaced = V[handle_idxs].clone()
    displaced[:, 0] += 0.5
    manager.set_handle_constraints(handle_idxs, displaced)
    manager.iterate()

    V_deformed_before = manager.V_deformed.clone()
    manager.set_rest_state()

    assert torch.allclose(manager.V_rest, V_deformed_before)


def test_set_rest_state_rebuilds_laplacian(mesh):
    """After set_rest_state, L should reflect the new rest geometry."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    L_original = manager.L.to_dense().clone()

    handle_idxs = [0]
    displaced = V[handle_idxs].clone()
    displaced[:, 0] += 0.5
    manager.set_handle_constraints(handle_idxs, displaced)
    manager.iterate()
    manager.set_rest_state()

    assert not torch.allclose(manager.L.to_dense(), L_original)


def test_set_handle_constraints_rebuilds_solver(mesh):
    """Changing handle indices should rebuild the solver."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    manager.set_handle_constraints([0], V[[0]])
    solver_first = manager.solver

    manager.set_handle_constraints([1, 2], V[[1, 2]])
    assert manager.solver is not solver_first


def test_set_handle_constraints_same_indices_keeps_solver(mesh):
    """Setting the same handle indices should not rebuild the solver."""
    V, F, L = mesh
    manager = ARAPManager(V, F, device=V.device)
    manager.set_handle_constraints([0], V[[0]])
    solver_first = manager.solver

    manager.set_handle_constraints([0], V[[0]] + 0.1)
    assert manager.solver is solver_first
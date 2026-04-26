#!/usr/bin/env python3
"""
ARAP triangle mesh UI: display mesh, right-click to lock vertices (control points),
left-click to select a locked vertex (shows gizmo). Points are moved only via the
gizmo. Uses Polyscope for display and vertex selection; ImGui for input.
"""
import argparse
import sys
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import igl
import torch

from .arap import ARAPManager
from .steklov import ARAPManagerSteklov


MESH_NAME = "mesh"
VERTICES_POINT_CLOUD_NAME = "mesh_vertices"
LOCK_SCALAR_NAME = "locked"  # vertex scalar: 0 = free, 1 = locked
POINT_STATE_NAME = "point_state"  # 0=normal, 1=locked, 2=hovered, 3=locked+hovered
SELECTED_GIZMO_NAME = "selected_lock_gizmo"


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load triangle mesh from path; return (V, F) as numpy arrays."""
    V, F = igl.read_triangle_mesh(path)
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int32)


def remove_selected_gizmo() -> None:
    """Remove the single selected-vertex gizmo if it exists."""
    try:
        ps.remove_transformation_gizmo(SELECTED_GIZMO_NAME)
    except Exception:
        pass


def pick_vertex(screen_coords: tuple[float, float]) -> int | None:
    """Pick a vertex under screen coords from the vertex point cloud; return vertex index or None."""
    pick_result = ps.pick(screen_coords=screen_coords)
    hit = getattr(pick_result, "is_hit", getattr(pick_result, "isHit", False))
    if not hit or getattr(pick_result, "structure_name", None) != VERTICES_POINT_CLOUD_NAME:
        return None
    if not hasattr(pick_result, "local_index"):
        return None
    return int(pick_result.local_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARAP mesh UI: right-click lock vertex, left-click select (gizmo); move via gizmo only"
    )
    parser.add_argument(
        "mesh_path",
        nargs="?",
        default="meshes/armadillo.obj",
        help="Path to triangle mesh (.obj)",
    )
    parser.add_argument(
        "--steklov",
        action="store_true",
        help="Use Steklov-DtN ARAP variant (slower to initialize, more robust to large deformations)",
    )
    args = parser.parse_args()

    # UI state
    locked_vertices: set[int] = set()
    selected_locked_vertex: int | None = None  # only this locked vertex shows a gizmo
    first_frame: bool = True
    pause_arap: bool = True
    freeze_on_unanchoring: bool = False
    arap_iterations: int = 1

    V, F = load_mesh(args.mesh_path)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    n_vertices = V.shape[0]

    # arap state manager
    # arap = ARAPManager(V, F, device="cuda", float_dtype=torch.float32)
    if args.steklov:
        arap = ARAPManagerSteklov(V, F, device="cuda", float_dtype=torch.float32)
    else:
        arap = ARAPManager(V, F, device="cuda", float_dtype=torch.float32)

    ps.init()
    ps.set_automatically_compute_scene_extents(True)

    # Mesh with faces and edges; vertex selection for picking
    mesh_handle = ps.register_surface_mesh(
        MESH_NAME,
        V,
        F,
        edge_width=1.0,
        edge_color=(0.25, 0.25, 0.2),
        smooth_shade=True,
    )
    mesh_handle.set_selection_mode("vertices_only")

    # Lock state: 0 = free, 1 = locked (locked vertices drawn in red via cmap)
    lock_values = np.zeros(n_vertices, dtype=np.int32)
    mesh_handle.add_scalar_quantity(
        LOCK_SCALAR_NAME,
        lock_values,
        enabled=True,
        cmap="turbo",
        datatype="categorical",
        vminmax=(0, 4),
    )

    # Point cloud at vertices for visibility; same index as mesh vertices
    # vertex_radius = float(np.ptp(V)) * 0.002
    pts_handle = ps.register_point_cloud(
        VERTICES_POINT_CLOUD_NAME,
        V,
        radius=0.01,
        color=(0.9, 0.85, 0.7),
    )
    # Point cloud state: 0=normal, 1=locked, 2=hovered, 3=locked+hovered (updated each frame)
    point_state = np.zeros(n_vertices, dtype=np.float64)
    pts_handle.add_scalar_quantity(
        POINT_STATE_NAME,
        point_state,
        enabled=True,
        cmap="turbo",
        datatype="categorical",
        vminmax=(0, 3),
    )

    def update_arap_handles():
        nonlocal arap, locked_vertices
        locked_vertices_list = list(locked_vertices)
        arap.set_handle_constraints(
            locked_vertices_list,
            V[locked_vertices_list],
        )

    def run_arap_iterations(num_iters: int) -> None:
        nonlocal V
        if len(locked_vertices) == 0:
            return

        for _ in range(num_iters):
            arap.iterate()

        V = arap.V_deformed.cpu().numpy()
        mesh_handle.update_vertex_positions(V)
        pts_handle.update_point_positions(V)

    def reset_mesh_to_rest_state() -> None:
        nonlocal V, selected_locked_vertex
        V = arap.reset_to_rest_state().detach().cpu().numpy().copy()
        mesh_handle.update_vertex_positions(V)
        pts_handle.update_point_positions(V)

        if selected_locked_vertex is not None and 0 <= selected_locked_vertex < n_vertices:
            try:
                g = ps.get_transformation_gizmo(SELECTED_GIZMO_NAME)
                g.set_position(V[selected_locked_vertex].copy())
            except Exception:
                pass

    def user_callback() -> None:
        nonlocal locked_vertices, selected_locked_vertex, V, first_frame
        nonlocal pause_arap, arap_iterations, freeze_on_unanchoring
        if first_frame:
            psim.SetWindowCollapsed("Polyscope", True)
            first_frame = False
        io = psim.GetIO()
        mouse_pos = (float(io.MousePos[0]), float(io.MousePos[1]))
        ui_capturing_mouse = getattr(io, "WantCaptureMouse", False)

        _, pause_arap = psim.Checkbox("Pause ARAP", pause_arap)
        _, arap_iterations = psim.InputInt("ARAP Iterations", arap_iterations, step=1, step_fast=10)
        arap_iterations = max(1, arap_iterations)
        _, freeze_on_unanchoring = psim.Checkbox("Reset rest state on unanchoring", freeze_on_unanchoring)
        manual_iterate = psim.Button("Iterate ARAP")
        reset_mesh = psim.Button("Reset Mesh to Rest State")

        # Suppress default camera movement when a gizmo (or other UI) is using the mouse
        # if getattr(io, "WantCaptureMouse", False):
        #     ps.set_do_default_mouse_interaction(False)
        # else:
        #     ps.set_do_default_mouse_interaction(True)

        # Update lock-state scalar on mesh
        lock_values = np.zeros(n_vertices, dtype=np.int32)
        for i in locked_vertices:
            if 0 <= i < n_vertices:
                lock_values[i] = 1
        mesh_handle.add_scalar_quantity(
            LOCK_SCALAR_NAME,
            lock_values,
            datatype="categorical",
        )

        # Point cloud state: 0=normal, 1=locked, 2=hovered, 3=locked+hovered
        hovered_vertex = pick_vertex(mouse_pos)
        point_state = np.zeros(n_vertices, dtype=np.float64)
        for i in locked_vertices:
            if 0 <= i < n_vertices:
                point_state[i] = 1.0
        if hovered_vertex is not None and 0 <= hovered_vertex < n_vertices:
            point_state[hovered_vertex] += 2.0  # 2=hovered only, 3=locked+hovered
        pts_handle.add_scalar_quantity(
            POINT_STATE_NAME,
            point_state,
            enabled=True,
            cmap="viridis",
            datatype="categorical",
            vminmax=(0, 3),
        )

        # ---- Sync vertex position from selected gizmo (gizmo -> vertex) ----
        if selected_locked_vertex is not None and 0 <= selected_locked_vertex < n_vertices:
            try:
                g = ps.get_transformation_gizmo(SELECTED_GIZMO_NAME)
                pos = np.asarray(g.get_position(), dtype=np.float64).reshape(-1)[:3]
                if pos.size >= 3:
                    V[selected_locked_vertex] = pos
                    mesh_handle.update_vertex_positions(V)
                    pts_handle.update_point_positions(V)

                    arap.set_handle_positions(V[list(locked_vertices)])
            except Exception:
                pass

        # ---- Right click: lock/unlock vertex under cursor ----
        if not ui_capturing_mouse and psim.IsMouseClicked(1):
            idx = pick_vertex(mouse_pos)
            if idx is not None and 0 <= idx < n_vertices:
                if idx in locked_vertices:
                    locked_vertices.discard(idx)
                    if idx == selected_locked_vertex:
                        remove_selected_gizmo()
                        selected_locked_vertex = None
                    if freeze_on_unanchoring:
                        arap.set_rest_state()
                else:
                    locked_vertices.add(idx)
                    remove_selected_gizmo()
                    selected_locked_vertex = idx
                    gizmo = ps.add_transformation_gizmo(SELECTED_GIZMO_NAME)
                    gizmo.set_allow_rotation(False)
                    gizmo.set_allow_scaling(False)
                    gizmo.set_position(V[idx].copy())
                
                update_arap_handles()
            return

        # ---- Left click: select locked vertex (show gizmo) or clear selection ----
        if not ui_capturing_mouse and psim.IsMouseClicked(0):
            idx = pick_vertex(mouse_pos)
            if idx is not None and 0 <= idx < n_vertices and idx in locked_vertices:
                remove_selected_gizmo()
                selected_locked_vertex = idx
                gizmo = ps.add_transformation_gizmo(SELECTED_GIZMO_NAME)
                gizmo.set_allow_rotation(False)
                gizmo.set_position(V[idx].copy())
            else:
                # Don't clear selection if a gizmo handle (or other UI) is capturing the click
                if getattr(io, "WantCaptureMouse", False):
                    return
                remove_selected_gizmo()
                selected_locked_vertex = None
            return

        # ---- Update ARAP -----
        # if psim.IsMouseReleased(0):
        #     arap.set_rest_state()
        if reset_mesh:
            reset_mesh_to_rest_state()
        elif manual_iterate:
            run_arap_iterations(arap_iterations)
        elif not pause_arap:
            run_arap_iterations(arap_iterations)

    ps.set_user_callback(user_callback)
    ps.show()


if __name__ == "__main__":
    main()

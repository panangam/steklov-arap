import json
import numpy as np
import igl
import trimesh
from matplotlib import colors as mcolors

# Load mesh with libigl (preserves original vertex/face ordering)
V, F = igl.read_triangle_mesh("../meshes/golem.obj")

# Wrap in trimesh; process=False keeps ordering intact
mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

# Find connected components via face adjacency
components = trimesh.graph.connected_components(
    edges=mesh.face_adjacency,
    nodes=np.arange(len(mesh.faces)),
)

# Pick visually distinct named colors from matplotlib
color_names = [
    "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
    "tab:cyan", "tab:pink", "tab:olive", "tab:brown", "tab:gray",
    "gold", "magenta", "lime", "navy", "teal",
]
if len(components) > len(color_names):
    raise ValueError(
        f"Found {len(components)} components but only {len(color_names)} colors available"
    )

# Build {name: vertex_indices} dict and assign vertex colors
component_data = {}
vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
vertex_colors[:, 3] = 255  # default alpha

for face_idx, name in zip(components, color_names):
    vert_idx = np.unique(mesh.faces[face_idx].flatten())
    component_data[name] = vert_idx.tolist()

    rgba = (np.array(mcolors.to_rgba(name)) * 255).astype(np.uint8)
    vertex_colors[vert_idx] = rgba

# Write JSON
with open("components.json", "w") as f:
    json.dump(component_data, f, indent=2)

print(f"Wrote {len(component_data)} components to components.json")
for name, idx in component_data.items():
    print(f"  {name}: {len(idx)} vertices")

# Set vertex colors and show
mesh.visual.vertex_colors = vertex_colors
mesh.show()
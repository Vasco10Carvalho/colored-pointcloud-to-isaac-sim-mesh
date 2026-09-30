#!/usr/bin/env python3
"""Prepare a coloured PLY for a portable Isaac Sim environment.

The script never modifies the source.  It creates a local-coordinate clean cloud,
a Poisson visual mesh, an independent coarse collision mesh, GLB interchange
assets, a point/splat NPZ alternative, previews, and machine-readable metrics.
USD authoring is deliberately a separate step because it uses Isaac Sim Python.

When voxel sizes or Poisson depths are set to zero, conservative values are
derived from the measured nearest-neighbour spacing instead of using a fixed
value that only suits one scanner or scene.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import struct
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def parse_ply_header(path: Path) -> dict:
    raw = bytearray()
    with path.open("rb") as stream:
        while not raw.endswith(b"end_header\n"):
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header incomplete: {path}")
            raw.extend(line)
    text = raw.decode("ascii", errors="replace")
    vertex_count = None
    properties = []
    in_vertex = False
    for line in text.splitlines():
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
            in_vertex = True
        elif fields[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and fields[:1] == ["property"]:
            properties.append(fields[1:])
    return {
        "format": next((x for x in text.splitlines() if x.startswith("format ")), "unknown"),
        "header_bytes": len(raw),
        "vertex_count": vertex_count,
        "vertex_properties": properties,
    }


def estimate_nn(pcd: o3d.geometry.PointCloud, sample_size: int = 20_000) -> dict:
    points = np.asarray(pcd.points)
    if len(points) < 2:
        return {}
    rng = np.random.default_rng(20260923)
    chosen = rng.choice(len(points), min(sample_size, len(points)), replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []
    for idx in chosen:
        count, _, dist2 = tree.search_knn_vector_3d(pcd.points[int(idx)], 2)
        if count > 1:
            distances.append(math.sqrt(dist2[1]))
    values = np.asarray(distances)
    return {
        "sample_size": len(values),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "zero_distance_samples": int(np.count_nonzero(values == 0)),
    }


def bbox(points: np.ndarray) -> dict:
    low = points.min(axis=0)
    high = points.max(axis=0)
    return {
        "min": low.tolist(),
        "max": high.tolist(),
        "dimensions": (high - low).tolist(),
    }


def cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def remove_tiny_components(
    mesh: o3d.geometry.TriangleMesh, min_triangles: int, min_area: float
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    labels, counts, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    areas = np.asarray(areas)
    if not len(counts):
        return mesh, {"components_before": 0, "components_removed": 0}
    discard_components = np.where((counts < min_triangles) & (areas < min_area))[0]
    remove = np.isin(labels, discard_components)
    removed_triangles = int(remove.sum())
    if removed_triangles:
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
    return mesh, {
        "components_before": int(len(counts)),
        "components_removed": int(len(discard_components)),
        "triangles_removed": removed_triangles,
        "largest_component_triangle_fraction": float(counts.max() / max(1, counts.sum())),
    }


def supported_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int,
    scale: float,
    density_quantile: float,
    max_support_distance: float,
    bbox_padding: float,
    min_component_triangles: int,
    min_component_area: float,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=False, n_threads=-1
    )
    raw_counts = {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)}
    density = np.asarray(densities)
    vertices = np.asarray(mesh.vertices)
    cloud_points = np.asarray(pcd.points)
    support_tree = cKDTree(cloud_points)
    support_distance, _ = support_tree.query(vertices, workers=-1)
    density_threshold = float(np.quantile(density, density_quantile))
    low = cloud_points.min(axis=0) - bbox_padding
    high = cloud_points.max(axis=0) + bbox_padding
    remove = (
        (density < density_threshold)
        | (support_distance > max_support_distance)
        | np.any(vertices < low, axis=1)
        | np.any(vertices > high, axis=1)
    )
    removed_vertices = int(remove.sum())
    mesh.remove_vertices_by_mask(remove)
    mesh = cleanup_mesh(mesh)
    mesh, component_stats = remove_tiny_components(
        mesh, min_component_triangles, min_component_area
    )
    mesh = cleanup_mesh(mesh)
    nonmanifold = len(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    open_or_nonmanifold = len(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    return mesh, {
        "method": "poisson",
        "depth": depth,
        "scale": scale,
        "density_trim_quantile": density_quantile,
        "density_threshold": density_threshold,
        "max_support_distance_m": max_support_distance,
        "raw": raw_counts,
        "vertices_removed_by_density_support_bbox": removed_vertices,
        "components": component_stats,
        "final": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)},
        "boundary_or_nonmanifold_edges": open_or_nonmanifold,
        "nonmanifold_edges": nonmanifold,
        "boundary_edges_estimate": open_or_nonmanifold - nonmanifold,
        "elapsed_seconds": time.time() - start,
    }


def transfer_colors(pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh) -> None:
    if not pcd.has_colors() or not len(mesh.vertices):
        return
    tree = cKDTree(np.asarray(pcd.points))
    _, nearest = tree.query(np.asarray(mesh.vertices), workers=-1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[nearest])


def choose_processing_parameters(args, raw_stats: dict) -> dict:
    """Fill automatic parameters from measured point spacing and dataset size."""
    nn = raw_stats.get("nearest_neighbour_m") or {}
    spacing = float(nn.get("median") or 0.0)
    if spacing <= 0.0:
        spacing = float(nn.get("p90") or 0.0)
    if spacing <= 0.0:
        raise RuntimeError(
            "Could not estimate point spacing. Pass explicit voxel sizes instead."
        )

    chosen = {}
    if args.clean_voxel <= 0.0:
        args.clean_voxel = max(spacing * 6.0, 1e-5)
        chosen["clean_voxel"] = "6 x raw median nearest-neighbour spacing"
    if args.visual_voxel <= 0.0:
        args.visual_voxel = max(args.clean_voxel, spacing * 10.0)
        chosen["visual_voxel"] = "max(clean voxel, 10 x raw median spacing)"
    if args.collision_voxel <= 0.0:
        args.collision_voxel = max(args.visual_voxel * 8.0, spacing * 80.0)
        chosen["collision_voxel"] = "max(8 x visual voxel, 80 x raw median spacing)"
    if args.points_voxel <= 0.0:
        args.points_voxel = max(args.visual_voxel * 2.0, spacing * 20.0)
        chosen["points_voxel"] = "max(2 x visual voxel, 20 x raw median spacing)"
    if args.visual_depth <= 0:
        args.visual_depth = 12 if raw_stats["points_read"] >= 10_000_000 else 11
        chosen["visual_depth"] = "selected from input point count"
    if args.collision_depth <= 0:
        args.collision_depth = 9 if raw_stats["points_read"] >= 1_000_000 else 8
        chosen["collision_depth"] = "selected from input point count"
    return {
        "mode": "density-adaptive" if chosen else "explicit",
        "raw_spacing_used": spacing,
        "rules_applied": chosen,
    }


def write_glb(path: Path, mesh: o3d.geometry.TriangleMesh, include_colors: bool) -> None:
    """Write a compact glTF 2.0 binary with vertex colours and opaque material."""
    positions = np.ascontiguousarray(np.asarray(mesh.vertices), dtype="<f4")
    normals = np.ascontiguousarray(np.asarray(mesh.vertex_normals), dtype="<f4")
    indices = np.ascontiguousarray(np.asarray(mesh.triangles).reshape(-1), dtype="<u4")
    colors = None
    if include_colors and mesh.has_vertex_colors():
        colors = np.ascontiguousarray(
            np.clip(np.rint(np.asarray(mesh.vertex_colors) * 255.0), 0, 255), dtype=np.uint8
        )

    arrays = [("POSITION", positions), ("NORMAL", normals)]
    if colors is not None:
        arrays.append(("COLOR_0", colors))
    arrays.append(("INDICES", indices))
    views = []
    accessors = []
    offsets = {}
    binary_size = 0
    for name, array in arrays:
        binary_size = (binary_size + 3) & ~3
        offsets[name] = binary_size
        byte_length = int(array.nbytes)
        target = 34963 if name == "INDICES" else 34962
        views.append({"buffer": 0, "byteOffset": binary_size, "byteLength": byte_length, "target": target})
        view_index = len(views) - 1
        if name == "INDICES":
            accessors.append(
                {
                    "bufferView": view_index,
                    "componentType": 5125,
                    "count": int(array.size),
                    "type": "SCALAR",
                    "min": [int(array.min())],
                    "max": [int(array.max())],
                }
            )
        elif name == "COLOR_0":
            accessors.append(
                {
                    "bufferView": view_index,
                    "componentType": 5121,
                    "normalized": True,
                    "count": len(array),
                    "type": "VEC3",
                }
            )
        else:
            accessor = {
                "bufferView": view_index,
                "componentType": 5126,
                "count": len(array),
                "type": "VEC3",
            }
            if name == "POSITION":
                accessor["min"] = array.min(axis=0).astype(float).tolist()
                accessor["max"] = array.max(axis=0).astype(float).tolist()
            accessors.append(accessor)
        binary_size += byte_length
    binary_size = (binary_size + 3) & ~3

    accessor_by_name = {name: index for index, (name, _) in enumerate(arrays)}
    attributes = {
        "POSITION": accessor_by_name["POSITION"],
        "NORMAL": accessor_by_name["NORMAL"],
    }
    if colors is not None:
        attributes["COLOR_0"] = accessor_by_name["COLOR_0"]
    document = {
        "asset": {"version": "2.0", "generator": "isaacsim_ply_pipeline"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": path.stem}],
        "meshes": [
            {
                "name": "mesh",
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": accessor_by_name["INDICES"],
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "OpaqueVertexColor",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
                "alphaMode": "OPAQUE",
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": binary_size}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    total_size = 12 + 8 + len(json_bytes) + 8 + binary_size
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total_size))
        stream.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        stream.write(json_bytes)
        stream.write(struct.pack("<I4s", binary_size, b"BIN\x00"))
        cursor = 0
        for name, array in arrays:
            wanted = offsets[name]
            if wanted > cursor:
                stream.write(b"\x00" * (wanted - cursor))
                cursor = wanted
            array.tofile(stream)
            cursor += array.nbytes
        if cursor < binary_size:
            stream.write(b"\x00" * (binary_size - cursor))


def write_mesh_assets(
    folder: Path, stem: str, mesh: o3d.geometry.TriangleMesh, include_colors: bool
) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    ply = folder / f"{stem}.ply"
    obj = folder / f"{stem}.obj"
    glb = folder / f"{stem}.glb"
    arrays = folder / f"{stem}_arrays.npz"
    log(f"Writing {ply.name}")
    if not o3d.io.write_triangle_mesh(
        str(ply), mesh, write_ascii=False, write_vertex_normals=True, write_vertex_colors=include_colors
    ):
        raise RuntimeError(f"Failed to write {ply}")
    log(f"Writing {obj.name}")
    if not o3d.io.write_triangle_mesh(
        str(obj), mesh, write_ascii=True, write_vertex_normals=True, write_vertex_colors=include_colors
    ):
        raise RuntimeError(f"Failed to write {obj}")
    log(f"Writing {glb.name}")
    write_glb(glb, mesh, include_colors)
    payload = {
        "vertices": np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float32),
        "triangles": np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.int32),
        "normals": np.ascontiguousarray(np.asarray(mesh.vertex_normals), dtype=np.float32),
    }
    if include_colors and mesh.has_vertex_colors():
        payload["colors"] = np.ascontiguousarray(
            np.asarray(mesh.vertex_colors), dtype=np.float32
        )
    np.savez(arrays, **payload)
    return {"ply": ply.name, "obj": obj.name, "glb": glb.name, "arrays": arrays.name}


def make_previews(
    folder: Path,
    cloud: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else np.full((len(points), 3), 0.5)
    ids = rng.choice(len(points), min(500_000, len(points)), replace=False)
    p = points[ids]
    c = np.clip(colors[ids], 0, 1)
    outputs = []

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.scatter(p[:, 0], p[:, 1], c=c, s=0.18, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X local (m)")
    ax.set_ylabel("Y local (m)")
    ax.set_title("Point cloud limpa — vista superior RGB")
    ax.grid(alpha=0.15)
    out = folder / "pointcloud_clean_top.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    outputs.append(out.name)

    vertices = np.asarray(mesh.vertices)
    vertex_colors = (
        np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full((len(vertices), 3), 0.5)
    )
    ids = rng.choice(len(vertices), min(250_000, len(vertices)), replace=False)
    v = vertices[ids]
    vc = np.clip(vertex_colors[ids], 0, 1)
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=vc, s=0.15, linewidths=0, alpha=1.0)
    ax.view_init(elev=35, azim=-60)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Mesh visual — amostra de vértices RGB, opacidade 1")
    out = folder / "visual_mesh_oblique.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    outputs.append(out.name)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clean-voxel", type=float, default=0.0, help="0 chooses from measured density")
    parser.add_argument("--visual-voxel", type=float, default=0.0, help="0 chooses from measured density")
    parser.add_argument("--visual-depth", type=int, default=0, help="0 chooses from point count")
    parser.add_argument("--collision-voxel", type=float, default=0.0, help="0 chooses from measured density")
    parser.add_argument("--collision-depth", type=int, default=0, help="0 chooses from point count")
    parser.add_argument("--collision-triangles", type=int, default=80_000)
    parser.add_argument("--points-voxel", type=float, default=0.0, help="0 chooses from measured density")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    root = args.output.expanduser().resolve()
    for relative in ("visual", "collision", "pointcloud", "scripts", "reports/previews"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), root / "scripts" / Path(__file__).name)

    if not source.is_file():
        raise FileNotFoundError(source)

    report = {
        "source_file_name": source.name,
        "source_size_bytes": source.stat().st_size,
        "ply_header": parse_ply_header(source),
        "coordinate_system": {
            "inference": "Input is treated as metres; verify against a known measurement",
            "vertical_axis": "Z",
            "scene_up_axis": "Z",
            "meters_per_unit": 1.0,
            "rotation_applied_degrees_xyz": [0.0, 0.0, 0.0],
        },
        "parameters": {"input": source.name, "output": "."},
    }

    start = time.time()
    log(f"Reading source {source}")
    raw = o3d.io.read_point_cloud(str(source), print_progress=True)
    raw_points = np.asarray(raw.points)
    if not len(raw_points):
        raise RuntimeError("Open3D did not read any points")
    if not raw.has_colors():
        raise RuntimeError(
            "The input PLY has no RGB colours. This pipeline requires a coloured point cloud."
        )
    finite = np.isfinite(raw_points).all(axis=1)
    raw_bbox = bbox(raw_points[finite])
    raw_stats = {
        "points_read": len(raw_points),
        "invalid_xyz": int((~finite).sum()),
        "bbox_utm": raw_bbox,
        "has_rgb": raw.has_colors(),
        "rgb_min_0_1": np.asarray(raw.colors).min(axis=0).tolist() if raw.has_colors() else None,
        "rgb_max_0_1": np.asarray(raw.colors).max(axis=0).tolist() if raw.has_colors() else None,
        "nearest_neighbour_m": estimate_nn(raw),
    }
    report["raw"] = raw_stats
    report["parameter_selection"] = choose_processing_parameters(args, raw_stats)
    report["parameters"].update(
        {
            "clean_voxel": args.clean_voxel,
            "visual_voxel": args.visual_voxel,
            "visual_depth": args.visual_depth,
            "collision_voxel": args.collision_voxel,
            "collision_depth": args.collision_depth,
            "collision_triangles": args.collision_triangles,
            "points_voxel": args.points_voxel,
        }
    )
    origin = np.array(
        [
            (raw_bbox["min"][0] + raw_bbox["max"][0]) * 0.5,
            (raw_bbox["min"][1] + raw_bbox["max"][1]) * 0.5,
            raw_bbox["min"][2],
        ]
    )
    report["coordinate_system"]["local_origin_source_coordinates"] = origin.tolist()
    if not finite.all():
        raw.remove_non_finite_points()
    raw.translate(-origin)

    log(f"Voxel downsample clean cloud at {args.clean_voxel:.3f} m")
    clean_before = raw.voxel_down_sample(args.clean_voxel)
    del raw, raw_points
    gc.collect()
    before_filter = len(clean_before.points)
    log("Conservative statistical outlier filter: neighbours=16, std_ratio=4.0")
    clean, _ = clean_before.remove_statistical_outlier(nb_neighbors=16, std_ratio=4.0)
    del clean_before
    gc.collect()
    normal_radius = args.clean_voxel * 3.0
    clean.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60)
    )
    clean_points = np.asarray(clean.points)
    clean.orient_normals_towards_camera_location(
        np.array([0.0, 0.0, float(clean_points[:, 2].max() + 30.0)])
    )
    clean_path = root / "pointcloud" / "map_clean.ply"
    if not o3d.io.write_point_cloud(str(clean_path), clean, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {clean_path}")
    report["clean"] = {
        "voxel_m": args.clean_voxel,
        "points_before_outlier_filter": before_filter,
        "points_after_outlier_filter": len(clean.points),
        "outliers_removed": before_filter - len(clean.points),
        "outliers_removed_fraction": (before_filter - len(clean.points)) / before_filter,
        "statistical_filter": {"neighbours": 16, "std_ratio": 4.0},
        "normal_radius_m": normal_radius,
        "normal_max_nn": 60,
        "normal_orientation": "towards virtual acquisition point 30 m above local max Z",
        "bbox_local": bbox(np.asarray(clean.points)),
        "nearest_neighbour_m": estimate_nn(clean),
        "colors_present": clean.has_colors(),
        "normals_present": clean.has_normals(),
        "path": "pointcloud/map_clean.ply",
        "deduplication": "Exact duplicates and sub-voxel repeats collapse during voxel aggregation",
    }

    log(f"Preparing visual meshing cloud at {args.visual_voxel:.3f} m")
    visual_cloud = clean.voxel_down_sample(args.visual_voxel)
    visual_radius = args.visual_voxel * 3.0
    visual_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=visual_radius, max_nn=60)
    )
    visual_cloud.orient_normals_towards_camera_location(
        np.array([0.0, 0.0, float(np.asarray(visual_cloud.points)[:, 2].max() + 30.0)])
    )
    np.savez_compressed(
        root / "visual" / "visual_points.npz",
        points=np.asarray(visual_cloud.points).astype(np.float32),
        colors=np.clip(np.rint(np.asarray(visual_cloud.colors) * 255.0), 0, 255).astype(np.uint8),
    )
    splat_voxel = args.points_voxel
    splat_cloud = visual_cloud.voxel_down_sample(splat_voxel)
    np.savez_compressed(
        root / "visual" / "visual_points_splats.npz",
        points=np.asarray(splat_cloud.points).astype(np.float32),
        colors=np.clip(np.rint(np.asarray(splat_cloud.colors) * 255.0), 0, 255).astype(np.uint8),
    )
    report["visual_point_alternative"] = {
        "points": len(splat_cloud.points),
        "voxel_m": splat_voxel,
        "suggested_width_m": 0.09,
        "reason": "Points preserve foliage, poles and fine disconnected structures that Poisson can smooth or bridge",
    }

    log(f"Poisson visual reconstruction depth={args.visual_depth}")
    visual_mesh, visual_stats = supported_poisson(
        visual_cloud,
        depth=args.visual_depth,
        scale=1.05,
        density_quantile=0.005,
        max_support_distance=args.visual_voxel * 3.0,
        bbox_padding=args.visual_voxel,
        min_component_triangles=20,
        min_component_area=0.01,
    )
    log("Transferring nearest RGB values to visual mesh vertices")
    transfer_colors(visual_cloud, visual_mesh)
    visual_assets = write_mesh_assets(root / "visual", "visual_mesh", visual_mesh, True)
    visual_stats.update(
        {
            "input_points": len(visual_cloud.points),
            "input_voxel_m": args.visual_voxel,
            "normal_radius_m": visual_radius,
            "colors_present": visual_mesh.has_vertex_colors(),
            "bbox_local": bbox(np.asarray(visual_mesh.vertices)),
            "assets": visual_assets,
            "method_selection": {
                "selected": "poisson",
                "reason": "Poisson is the robust default for dense, noisy environmental scans; compare BPA separately when required",
            },
        }
    )
    report["visual_mesh"] = visual_stats

    log(f"Preparing independent collision cloud at {args.collision_voxel:.3f} m")
    collision_cloud = clean.voxel_down_sample(args.collision_voxel)
    collision_before = len(collision_cloud.points)
    collision_cloud, _ = collision_cloud.remove_statistical_outlier(
        nb_neighbors=10, std_ratio=2.5
    )
    collision_radius = args.collision_voxel * 3.0
    collision_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=collision_radius, max_nn=60)
    )
    collision_cloud.orient_normals_towards_camera_location(
        np.array([0.0, 0.0, float(np.asarray(collision_cloud.points)[:, 2].max() + 30.0)])
    )
    collision_input_path = root / "collision" / "collision_input.ply"
    o3d.io.write_point_cloud(str(collision_input_path), collision_cloud, write_ascii=False)
    log(f"Poisson collision reconstruction depth={args.collision_depth}")
    collision_mesh, collision_stats = supported_poisson(
        collision_cloud,
        depth=args.collision_depth,
        scale=1.05,
        density_quantile=0.02,
        max_support_distance=args.collision_voxel * 3.0,
        bbox_padding=args.collision_voxel,
        min_component_triangles=50,
        min_component_area=1.0,
    )
    pre_simplify = len(collision_mesh.triangles)
    if pre_simplify > args.collision_triangles:
        collision_mesh = collision_mesh.simplify_quadric_decimation(args.collision_triangles)
        collision_mesh = cleanup_mesh(collision_mesh)
    collision_mesh.vertex_colors = o3d.utility.Vector3dVector()
    collision_assets = write_mesh_assets(
        root / "collision", "collision_mesh", collision_mesh, False
    )
    collision_stats.update(
        {
            "input_points_before_filter": collision_before,
            "input_points_after_filter": len(collision_cloud.points),
            "input_voxel_m": args.collision_voxel,
            "statistical_filter": {"neighbours": 10, "std_ratio": 2.5},
            "normal_radius_m": collision_radius,
            "triangles_before_simplification": pre_simplify,
            "target_triangles": args.collision_triangles,
            "final_vertices": len(collision_mesh.vertices),
            "final_triangles": len(collision_mesh.triangles),
            "bbox_local": bbox(np.asarray(collision_mesh.vertices)),
            "assets": collision_assets,
            "derived_independently_from_reduced_point_cloud": True,
        }
    )
    report["collision_mesh"] = collision_stats

    log("Generating preview images")
    report["previews"] = make_previews(root / "reports" / "previews", clean, visual_mesh)
    report["elapsed_processing_seconds"] = time.time() - start
    report_path = root / "reports" / "processing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Processing report: {report_path}")
    log(f"Point-cloud and mesh phase finished in {report['elapsed_processing_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

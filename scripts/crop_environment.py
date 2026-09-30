#!/usr/bin/env python3
"""Crop a processed environment to an axis-aligned interior cutaway.

This script uses the already-cleaned cloud and reconstructed meshes.  It does
not alter the original source PLY.  Bounds are expressed in the local Z-up,
metre coordinate system documented in processing_report.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def log(message: str) -> None:
    print(f"[interior] {message}", flush=True)


def load_pipeline_helpers(script_dir: Path):
    path = script_dir / "process_pointcloud.py"
    spec = importlib.util.spec_from_file_location("pointcloud_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pipeline helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point_mask(points: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    return (
        (points[:, 0] >= bounds["x_min"])
        & (points[:, 0] <= bounds["x_max"])
        & (points[:, 1] >= bounds["y_min"])
        & (points[:, 1] <= bounds["y_max"])
        & (points[:, 2] >= bounds["z_min"])
        & (points[:, 2] <= bounds["z_max"])
    )


def array_bbox(points: np.ndarray) -> dict[str, list[float]]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    return {
        "min": low.astype(float).tolist(),
        "max": high.astype(float).tolist(),
        "dimensions": (high - low).astype(float).tolist(),
    }


def crop_cloud(cloud: o3d.geometry.PointCloud, bounds: dict[str, float]) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    ids = np.flatnonzero(point_mask(points, bounds))
    return cloud.select_by_index(ids.tolist())


def crop_mesh(mesh: o3d.geometry.TriangleMesh, bounds: dict[str, float]) -> tuple[o3d.geometry.TriangleMesh, dict]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    inside = point_mask(vertices, bounds)
    keep = np.all(inside[triangles], axis=1)
    before = {"vertices": int(len(vertices)), "triangles": int(len(triangles))}
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    after = {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "bbox_local": array_bbox(np.asarray(mesh.vertices)),
    }
    return mesh, {"before": before, "after": after}


def save_mesh_arrays(path: Path, mesh: o3d.geometry.TriangleMesh, include_colors: bool) -> None:
    arrays: dict[str, np.ndarray] = {
        "vertices": np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float32),
        "triangles": np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.int32),
        "normals": np.ascontiguousarray(np.asarray(mesh.vertex_normals), dtype=np.float32),
    }
    if include_colors and mesh.has_vertex_colors():
        arrays["colors"] = np.ascontiguousarray(
            np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0), dtype=np.float32
        )
    np.savez(path, **arrays)


def estimate_floor_z(points: np.ndarray) -> float:
    z = points[:, 2]
    low = float(np.quantile(z, 0.01))
    high = float(np.quantile(z, 0.45))
    candidates = z[(z >= low) & (z <= high)]
    bin_width = max((high - low) / 500.0, 0.005)
    edges = np.arange(low, high + bin_width, bin_width)
    if len(edges) < 2:
        return float(np.median(candidates))
    counts, edges = np.histogram(candidates, bins=edges)
    index = int(np.argmax(counts))
    return float((edges[index] + edges[index + 1]) * 0.5)


def make_previews(folder: Path, cloud: o3d.geometry.PointCloud) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    rng = np.random.default_rng(44)
    ids = np.arange(len(points))
    if len(ids) > 450_000:
        ids = rng.choice(ids, 450_000, replace=False)
    sample = points[ids]
    sample_colors = colors[ids]

    top = folder / "interior_top.png"
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.scatter(sample[:, 0], sample[:, 1], s=0.18, c=sample_colors, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("X local (m)")
    ax.set_ylabel("Y local (m)")
    ax.set_title("Interior cutaway — RGB top view")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(top)
    plt.close(fig)

    oblique = folder / "interior_oblique.png"
    rng.shuffle(ids)
    ids = ids[: min(250_000, len(ids))]
    sample = points[ids]
    sample_colors = colors[ids]
    fig = plt.figure(figsize=(13, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        sample[:, 0], sample[:, 1], sample[:, 2],
        s=0.08, c=sample_colors, depthshade=False, linewidths=0,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Interior cutaway — RGB oblique view")
    ax.view_init(elev=38, azim=-62)
    fig.tight_layout()
    fig.savefig(oblique)
    plt.close(fig)
    return [top.name, oblique.name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--point-voxel", type=float, default=0.10)
    parser.add_argument("--minimum-points", type=int, default=1000)
    args = parser.parse_args()
    root = args.root.resolve()
    helpers = load_pipeline_helpers(root / "scripts")
    bounds = {
        "x_min": args.x_min, "x_max": args.x_max,
        "y_min": args.y_min, "y_max": args.y_max,
        "z_min": args.z_min, "z_max": args.z_max,
    }

    source_cloud_path = root / "pointcloud" / "map_clean.ply"
    source_visual_path = root / "visual" / "visual_mesh.ply"
    source_collision_path = root / "collision" / "collision_mesh.ply"
    for path in (source_cloud_path, source_visual_path, source_collision_path):
        if not path.exists():
            raise FileNotFoundError(path)

    log("Loading clean point cloud")
    source_cloud = o3d.io.read_point_cloud(str(source_cloud_path))
    interior_cloud = crop_cloud(source_cloud, bounds)
    if len(interior_cloud.points) < args.minimum_points:
        raise RuntimeError("Interior crop is unexpectedly small")
    interior_cloud_path = root / "pointcloud" / "map_clean.ply"
    o3d.io.write_point_cloud(str(interior_cloud_path), interior_cloud, write_ascii=False)
    floor_z = estimate_floor_z(np.asarray(interior_cloud.points))
    log(f"Interior cloud: {len(interior_cloud.points):,} points; floor Z={floor_z:.3f}")

    point_cloud = interior_cloud.voxel_down_sample(args.point_voxel)
    point_arrays = root / "visual" / "visual_points_splats.npz"
    np.savez(
        point_arrays,
        points=np.ascontiguousarray(np.asarray(point_cloud.points), dtype=np.float32),
        colors=np.ascontiguousarray(np.asarray(point_cloud.colors), dtype=np.float32),
    )

    log("Cropping visual mesh")
    visual_mesh = o3d.io.read_triangle_mesh(str(source_visual_path), enable_post_processing=False)
    visual_mesh, visual_stats = crop_mesh(visual_mesh, bounds)
    if not visual_mesh.has_vertex_colors():
        raise RuntimeError("Visual mesh lost its vertex colours")
    visual_assets = helpers.write_mesh_assets(
        root / "visual", "visual_mesh", visual_mesh, True
    )
    visual_arrays = root / "visual" / "visual_mesh_arrays.npz"
    save_mesh_arrays(visual_arrays, visual_mesh, True)
    log(
        f"Visual interior: {len(visual_mesh.vertices):,} vertices, "
        f"{len(visual_mesh.triangles):,} triangles"
    )

    log("Cropping collision mesh")
    collision_mesh = o3d.io.read_triangle_mesh(str(source_collision_path), enable_post_processing=False)
    collision_mesh, collision_stats = crop_mesh(collision_mesh, bounds)
    collision_mesh.vertex_colors = o3d.utility.Vector3dVector()
    collision_assets = helpers.write_mesh_assets(
        root / "collision", "collision_mesh", collision_mesh, False
    )
    collision_arrays = root / "collision" / "collision_mesh_arrays.npz"
    save_mesh_arrays(collision_arrays, collision_mesh, False)
    log(
        f"Collision interior: {len(collision_mesh.vertices):,} vertices, "
        f"{len(collision_mesh.triangles):,} triangles"
    )

    previews = make_previews(root / "reports" / "previews", interior_cloud)
    stats = {
        "purpose": "axis-aligned interior cutaway; geometry outside the requested bounds removed",
        "bounds_local_m": bounds,
        "ceiling_strategy": f"remove all geometry above local Z={args.z_max} m",
        "floor_z_m": floor_z,
        "source_clean_points": int(len(source_cloud.points)),
        "interior_clean_points": int(len(interior_cloud.points)),
        "interior_point_visual_points": int(len(point_cloud.points)),
        "interior_cloud_bbox": array_bbox(np.asarray(interior_cloud.points)),
        "visual_mesh": visual_stats,
        "collision_mesh": collision_stats,
        "visual_assets": visual_assets,
        "collision_assets": collision_assets,
        "array_assets": {
            "visual": f"visual/{visual_arrays.name}",
            "collision": f"collision/{collision_arrays.name}",
            "points": f"visual/{point_arrays.name}",
        },
        "previews": previews,
    }
    report_path = root / "reports" / "interior_refinement_report.json"
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

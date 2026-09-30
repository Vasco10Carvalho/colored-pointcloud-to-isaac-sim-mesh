#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"Nao foi possivel ler pontos de {path}")
    pcd.remove_non_finite_points()
    pcd.remove_duplicated_points()
    return pcd


def build_point_cloud(points: np.ndarray, colors: np.ndarray | None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main() -> int:
    parser = argparse.ArgumentParser(description="Junta varios PLYs num unico point cloud.")
    parser.add_argument("inputs", type=Path, nargs="+", help="PLYs de entrada")
    parser.add_argument("output", type=Path, help="PLY de saida")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Voxel downsampling apos merge")
    parser.add_argument("--center-xy", action="store_true", help="Recentra o bbox em XY na origem")
    parser.add_argument("--zero-min-z", action="store_true", help="Desloca para min(Z)=0")
    parser.add_argument(
        "--translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="Translacao final",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Opcional: guarda offsets e bbox num JSON",
    )
    args = parser.parse_args()

    clouds = [load_cloud(path) for path in args.inputs]
    all_have_colors = all(cloud.has_colors() for cloud in clouds)

    points = np.vstack([np.asarray(cloud.points) for cloud in clouds])
    colors = np.vstack([np.asarray(cloud.colors) for cloud in clouds]) if all_have_colors else None
    merged = build_point_cloud(points, colors)
    print(f"[INFO] Pontos apos merge bruto: {len(merged.points)}")
    print(f"[INFO] Todos os tiles tem RGB: {all_have_colors}")

    if args.voxel_size > 0.0:
        before = len(merged.points)
        merged = merged.voxel_down_sample(args.voxel_size)
        print(f"[INFO] Voxel downsample {args.voxel_size}: {before} -> {len(merged.points)}")

    metadata: dict[str, object] = {
        "inputs": [path.name for path in args.inputs],
        "voxel_size": args.voxel_size,
        "center_xy": args.center_xy,
        "zero_min_z": args.zero_min_z,
        "translate": list(args.translate),
    }

    points = np.asarray(merged.points)
    bbox_min_before = points.min(axis=0)
    bbox_max_before = points.max(axis=0)
    metadata["bbox_min_before"] = bbox_min_before.tolist()
    metadata["bbox_max_before"] = bbox_max_before.tolist()

    applied_translation = np.zeros(3, dtype=float)
    if args.center_xy:
        xy_center = (bbox_min_before[:2] + bbox_max_before[:2]) * 0.5
        shift = np.array([-xy_center[0], -xy_center[1], 0.0], dtype=float)
        merged.translate(shift)
        applied_translation += shift
        print(f"[INFO] Recentrado em XY: {shift}")

    if args.zero_min_z:
        points = np.asarray(merged.points)
        min_z = float(points[:, 2].min())
        shift = np.array([0.0, 0.0, -min_z], dtype=float)
        merged.translate(shift)
        applied_translation += shift
        print(f"[INFO] min(Z) ajustado para 0: {shift}")

    if np.any(np.asarray(args.translate) != 0.0):
        shift = np.asarray(args.translate, dtype=float)
        merged.translate(shift)
        applied_translation += shift
        print(f"[INFO] Translacao extra aplicada: {shift}")

    points = np.asarray(merged.points)
    bbox_min_after = points.min(axis=0)
    bbox_max_after = points.max(axis=0)
    metadata["bbox_min_after"] = bbox_min_after.tolist()
    metadata["bbox_max_after"] = bbox_max_after.tolist()
    metadata["applied_translation"] = applied_translation.tolist()

    ensure_parent(args.output)
    ok = o3d.io.write_point_cloud(str(args.output), merged, write_ascii=False, compressed=False)
    if not ok:
        print(f"[ERRO] Falha ao gravar {args.output}")
        return 2
    print(f"[INFO] Guardado em: {args.output}")
    print(f"[INFO] BBox final min: {bbox_min_after}")
    print(f"[INFO] BBox final max: {bbox_max_after}")

    if args.metadata_output is not None:
        ensure_parent(args.metadata_output)
        args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[INFO] Metadata guardada em: {args.metadata_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

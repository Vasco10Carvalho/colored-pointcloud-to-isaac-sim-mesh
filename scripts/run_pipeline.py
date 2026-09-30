#!/usr/bin/env python3
"""Run the complete coloured PLY to portable Isaac Sim USD pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROCESS_KEYS = (
    "clean_voxel",
    "visual_voxel",
    "visual_depth",
    "collision_voxel",
    "collision_depth",
    "collision_triangles",
    "points_voxel",
)
CROP_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "point_voxel")


def log(message: str) -> None:
    print(f"[runner] {message}", flush=True)


def run(command: list[str]) -> None:
    log("Running: " + " ".join(command))
    subprocess.run(command, check=True)


def load_profile(path: Path | None) -> dict:
    if path is None:
        return {
            "name": "automatic-density-based",
            "processing": {
                "clean_voxel": 0,
                "visual_voxel": 0,
                "visual_depth": 0,
                "collision_voxel": 0,
                "collision_depth": 0,
                "collision_triangles": 80000,
                "points_voxel": 0,
            },
            "point_width": 0.0,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("processing"), dict):
        raise ValueError("Profile must contain a 'processing' object")
    return data


def copy_runtime_scripts(script_dir: Path, output: Path) -> None:
    destination = output / "scripts"
    destination.mkdir(parents=True, exist_ok=True)
    for source in script_dir.glob("*.py"):
        shutil.copy2(source, destination / source.name)


def add_numeric_options(command: list[str], values: dict, allowed: tuple[str, ...]) -> None:
    for key in allowed:
        if key not in values:
            continue
        flag = "--" + key.replace("_", "-")
        command.extend((flag, str(values[key])))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one coloured PLY into a portable Isaac Sim environment."
    )
    parser.add_argument("input", type=Path, help="Source coloured PLY; never modified")
    parser.add_argument("output", type=Path, help="Output environment directory")
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional JSON profile. Without one, density-adaptive parameters are used.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow writing into an existing output directory without deleting it.",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() != ".ply":
        raise ValueError(f"Expected a .ply input, got {input_path.name}")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{output} is not empty. Choose another directory or pass --resume."
        )
    output.mkdir(parents=True, exist_ok=True)

    profile = load_profile(args.profile.expanduser().resolve() if args.profile else None)
    log(f"Profile: {profile.get('name', 'unnamed')}")

    process_command = [
        sys.executable,
        str(script_dir / "process_pointcloud.py"),
        str(input_path),
        str(output),
    ]
    add_numeric_options(process_command, profile["processing"], PROCESS_KEYS)
    run(process_command)

    copy_runtime_scripts(script_dir, output)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    portable_profile = dict(profile)
    portable_profile.pop("source_path", None)
    (reports / "processing_profile.json").write_text(
        json.dumps(portable_profile, indent=2), encoding="utf-8"
    )

    crop = profile.get("crop")
    if crop:
        crop_command = [
            sys.executable,
            str(output / "scripts" / "crop_environment.py"),
            "--root",
            str(output),
        ]
        add_numeric_options(crop_command, crop, CROP_KEYS)
        if "minimum_points" in crop:
            crop_command.extend(("--minimum-points", str(crop["minimum_points"])))
        run(crop_command)

    processing = json.loads(
        (reports / "processing_report.json").read_text(encoding="utf-8")
    )
    point_width = float(profile.get("point_width", 0.0))
    if point_width <= 0.0:
        point_width = float(processing["parameters"]["points_voxel"]) * 0.9

    geometry_script = output / "scripts" / "geometry_to_usd.py"
    conversions = [
        (
            output / "visual" / "visual_mesh_arrays.npz",
            output / "visual" / "visual_mesh.usd",
            ("--kind", "mesh", "--name", "VisualMesh", "--color"),
        ),
        (
            output / "collision" / "collision_mesh_arrays.npz",
            output / "collision" / "collision_mesh.usd",
            ("--kind", "mesh", "--name", "CollisionMesh"),
        ),
        (
            output / "visual" / "visual_points_splats.npz",
            output / "visual" / "visual_points.usd",
            (
                "--kind",
                "points",
                "--name",
                "VisualPoints",
                "--point-width",
                str(point_width),
            ),
        ),
    ]
    for source, destination, extra in conversions:
        if not source.is_file():
            raise FileNotFoundError(source)
        run(
            [
                sys.executable,
                str(geometry_script),
                str(source),
                str(destination),
                *extra,
            ]
        )

    run(
        [
            sys.executable,
            str(output / "scripts" / "finalize_environment.py"),
            str(output),
            "--point-width",
            str(point_width),
            "--skip-convert",
            "--skip-points",
        ]
    )

    final_stage = output / "environment_ready.usda"
    if not final_stage.is_file():
        raise RuntimeError(f"Final stage was not created: {final_stage}")
    log(f"Done. Open this file in Isaac Sim: {final_stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Author a compact coloured USD mesh or point asset from NumPy arrays.

Run with Isaac Sim's Kit Python plus the bundled NumPy and USD paths.  This
script intentionally does not start SimulationApp or the RTX renderer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def vertex_color_material(stage: Usd.Stage, parent: str, prim) -> None:
    looks = UsdGeom.Scope.Define(stage, f"{parent}/Looks")
    material = UsdShade.Material.Define(stage, f"{parent}/Looks/VertexColor")
    reader = UsdShade.Shader.Define(stage, f"{parent}/Looks/VertexColor/DisplayColorReader")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

    surface = UsdShade.Shader.Define(stage, f"{parent}/Looks/VertexColor/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader_output)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    surface_output = surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_output)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def write_mesh(data, output: Path, name: str, with_color: bool) -> dict:
    vertices = np.ascontiguousarray(data["vertices"], dtype=np.float32)
    triangles = np.ascontiguousarray(data["triangles"], dtype=np.int32)
    normals = np.ascontiguousarray(data["normals"], dtype=np.float32)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = f"/{name}"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(triangles.reshape(-1), dtype=np.int32))
    )
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    low = [float(value) for value in vertices.min(axis=0)]
    high = [float(value) for value in vertices.max(axis=0)]
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*low), Gf.Vec3f(*high)]))
    if with_color:
        colors = np.ascontiguousarray(data["colors"], dtype=np.float32)
        if float(colors.max(initial=0.0)) > 1.0:
            colors /= 255.0
        colors = np.clip(colors, 0.0, 1.0)
        color = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        color.Set(Vt.Vec3fArray.FromNumpy(colors))
        opacity = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant
        )
        opacity.Set(Vt.FloatArray([1.0]))
        vertex_color_material(stage, root_path, mesh.GetPrim())
    stage.GetRootLayer().Save()
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "colored": bool(with_color),
        "output_bytes": output.stat().st_size,
    }


def write_points(data, output: Path, name: str, point_width: float) -> dict:
    points = np.ascontiguousarray(data["points"], dtype=np.float32)
    colors = np.ascontiguousarray(data["colors"], dtype=np.float32)
    if float(colors.max(initial=0.0)) > 1.0:
        colors /= 255.0
    colors = np.clip(colors, 0.0, 1.0)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = f"/{name}"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    geom = UsdGeom.Points.Define(stage, f"{root_path}/points")
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    geom.CreateWidthsAttr(Vt.FloatArray([float(point_width)]))
    geom.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    color = UsdGeom.PrimvarsAPI(geom.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    )
    color.Set(Vt.Vec3fArray.FromNumpy(colors))
    opacity = UsdGeom.PrimvarsAPI(geom.GetPrim()).CreatePrimvar(
        "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant
    )
    opacity.Set(Vt.FloatArray([1.0]))
    low = [float(value) for value in points.min(axis=0)]
    high = [float(value) for value in points.max(axis=0)]
    geom.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*low), Gf.Vec3f(*high)]))
    vertex_color_material(stage, root_path, geom.GetPrim())
    stage.GetRootLayer().Save()
    return {
        "points": int(len(points)),
        "colored": True,
        "point_width_m": float(point_width),
        "output_bytes": output.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=("mesh", "points"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--point-width", type=float, default=0.09)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        if args.kind == "mesh":
            result = write_mesh(data, args.output, args.name, args.color)
        else:
            result = write_points(data, args.output, args.name, args.point_width)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

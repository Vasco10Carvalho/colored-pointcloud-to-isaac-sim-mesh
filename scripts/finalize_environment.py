#!/usr/bin/env python3
"""Convert prepared GLBs to USD, assemble the Isaac Sim stage, and validate it.

Run this script with Isaac Sim's python.sh, not the system Python.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np


def log(message: str) -> None:
    print(f"[finalize] {message}", flush=True)


def copy_self(root: Path) -> None:
    target = root / "scripts" / Path(__file__).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__).resolve(), target)


async def convert_asset(input_path: Path, output_path: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.single_mesh = True
    context.merge_all_meshes = True
    context.smooth_normals = False
    context.export_preview_surface = True
    context.use_meter_as_world_unit = True
    context.create_world_as_default_root_prim = False
    context.embed_textures = True

    def progress(current: float, total: float) -> None:
        if total and (current == total or int(current) % max(1, int(total // 10)) == 0):
            log(f"Converting {input_path.name}: {current:.0f}/{total:.0f}")

    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(input_path), str(output_path), progress, context
    )
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(
            f"Asset conversion failed for {input_path}: {task.get_error_message()} "
            f"(status {task.get_status()})"
        )


def mesh_prims(stage):
    from pxr import UsdGeom

    return [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]


def create_points_usd(npz_path: Path, usd_path: Path, width: float, chunk_size: int = 100_000) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    log(f"Creating point/splat USD from {npz_path.name}")
    with np.load(npz_path) as data:
        points = np.ascontiguousarray(data["points"], dtype=np.float32)
        colors = np.ascontiguousarray(data["colors"], dtype=np.float32) / 255.0
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/VisualPoints").GetPrim()
    stage.SetDefaultPrim(root)
    chunk_count = 0
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk_points = np.ascontiguousarray(points[start:stop])
        chunk_colors = np.ascontiguousarray(colors[start:stop])
        geom = UsdGeom.Points.Define(stage, f"/VisualPoints/chunk_{chunk_count:04d}")
        # Use the same authoring calls as the previously validated workspace utility.
        geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(chunk_points))
        geom.CreateWidthsAttr(Vt.FloatArray([width]))
        geom.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        primvars = UsdGeom.PrimvarsAPI(geom.GetPrim())
        display_color = primvars.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        display_color.Set(Vt.Vec3fArray.FromNumpy(chunk_colors))
        extent = [
            Gf.Vec3f(*chunk_points.min(axis=0)),
            Gf.Vec3f(*chunk_points.max(axis=0)),
        ]
        geom.CreateExtentAttr(Vt.Vec3fArray(extent))
        chunk_count += 1
        log(f"Authored point chunk {chunk_count}: {stop - start:,} points")
    stage.GetRootLayer().Save()
    if usd_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Point USD is unexpectedly small ({usd_path.stat().st_size} bytes): {usd_path}"
        )
    log(f"Point/splat USD written: {usd_path.stat().st_size / 1e6:.1f} MB")
    return {
        "points": int(len(points)),
        "width_m": width,
        "chunk_size": chunk_size,
        "chunks": chunk_count,
        "bbox_min": points.min(axis=0).astype(float).tolist(),
        "bbox_max": points.max(axis=0).astype(float).tolist(),
        "rgb_min": colors.min(axis=0).astype(float).tolist(),
        "rgb_max": colors.max(axis=0).astype(float).tolist(),
    }


def inspect_asset(path: Path) -> dict:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Could not open USD asset: {path}")
    meshes = mesh_prims(stage)
    details = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        color = mesh.GetDisplayColorPrimvar()
        opacity = mesh.GetDisplayOpacityPrimvar()
        details.append(
            {
                "path": str(prim.GetPath()),
                "points": len(mesh.GetPointsAttr().Get() or []),
                "faces": len(mesh.GetFaceVertexCountsAttr().Get() or []),
                "display_color": bool(color and color.HasValue()),
                "display_color_interpolation": str(color.GetInterpolation()) if color else "",
                "display_opacity": bool(opacity and opacity.HasValue()),
            }
        )
    default = stage.GetDefaultPrim()
    return {
        "path": path.name,
        "default_prim": str(default.GetPath()) if default else None,
        "mesh_count": len(meshes),
        "meshes": details,
    }


def add_reference(prim, stage_path: Path, asset_path: Path) -> None:
    relative = os.path.relpath(asset_path, stage_path.parent).replace(os.sep, "/")
    prim.GetReferences().AddReference(f"./{relative}")


def apply_visual_overrides(stage, root_path: str) -> list[str]:
    from pxr import UsdGeom, UsdPhysics, Vt

    paths = []
    root = stage.GetPrimAtPath(root_path)
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root.GetPath()) or not prim.IsA(UsdGeom.Mesh):
            continue
        paths.append(str(prim.GetPath()))
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(False)
        image = UsdGeom.Gprim(prim)
        opacity = image.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant)
        opacity.Set(Vt.FloatArray([1.0]))
    return paths


def apply_collision_overrides(stage, root_path: str) -> list[str]:
    from pxr import UsdGeom, UsdPhysics

    paths = []
    root = stage.GetPrimAtPath(root_path)
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root.GetPath()) or not prim.IsA(UsdGeom.Mesh):
            continue
        paths.append(str(prim.GetPath()))
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    return paths


def create_stage(
    stage_path: Path,
    visual_usd: Path,
    points_usd: Path,
    collision_usd: Path,
    processing: dict,
    show_points: bool,
) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    environment = UsdGeom.Xform.Define(stage, "/World/Environment").GetPrim()
    source_origin = processing["coordinate_system"]["local_origin_source_coordinates"]
    environment.SetCustomDataByKey(
        "sourceCoordinatesOrigin",
        ", ".join(f"{float(value):.12g}" for value in source_origin),
    )
    environment.SetCustomDataByKey("sourceVerticalAxis", "Z")
    environment.SetCustomDataByKey("sourceUnits", "metres")

    visual = UsdGeom.Xform.Define(stage, "/World/Environment/Visual").GetPrim()
    add_reference(visual, stage_path, visual_usd)
    UsdGeom.Imageable(visual).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible if show_points else UsdGeom.Tokens.inherited
    )

    points = UsdGeom.Xform.Define(stage, "/World/Environment/PointsAlternative").GetPrim()
    add_reference(points, stage_path, points_usd)
    UsdGeom.Imageable(points).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.inherited if show_points else UsdGeom.Tokens.invisible
    )

    collision = UsdGeom.Xform.Define(stage, "/World/Environment/Collision").GetPrim()
    add_reference(collision, stage_path, collision_usd)
    UsdGeom.Imageable(collision).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    clean_bbox = processing["clean"]["bbox_local"]
    low = np.asarray(clean_bbox["min"], dtype=float)
    high = np.asarray(clean_bbox["max"], dtype=float)
    padding = 1.0
    thickness = 0.20
    top = low[2] - 0.05
    center_z = top - thickness * 0.5
    ground = UsdGeom.Cube.Define(stage, "/World/Environment/GroundCollider")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(
        Gf.Vec3d(float((low[0] + high[0]) * 0.5), float((low[1] + high[1]) * 0.5), float(center_z))
    )
    ground.AddScaleOp().Set(
        Gf.Vec3f(float(high[0] - low[0] + 2 * padding), float(high[1] - low[1] + 2 * padding), thickness)
    )
    ground.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)

    # Save once so referenced descendants compose, then author explicit overrides.
    stage.GetRootLayer().Save()
    stage.Reload()
    visual_mesh_paths = apply_visual_overrides(stage, "/World/Environment/Visual")
    collision_mesh_paths = apply_collision_overrides(stage, "/World/Environment/Collision")
    stage.GetRootLayer().Save()
    return {
        "visual_mesh_paths": visual_mesh_paths,
        "collision_mesh_paths": collision_mesh_paths,
        "ground_top_z_m": float(top),
        "ground_thickness_m": thickness,
        "points_visible": show_points,
    }


def validate_stage(stage_path: Path, expect_points: bool = False) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    text = stage_path.read_text(encoding="utf-8")
    refs = re.findall(r"@([^@]+)@", text)
    absolute = [ref for ref in refs if Path(ref).is_absolute()]
    resolved = []
    missing = []
    for ref in refs:
        candidate = (stage_path.parent / ref).resolve()
        resolved.append({"reference": ref, "resolved": ref, "exists": candidate.exists()})
        if not candidate.exists():
            missing.append(ref)
    stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)
    if not stage:
        raise RuntimeError(f"Could not open assembled stage: {stage_path}")
    required = [
        "/World/Environment/Visual",
        "/World/Environment/Collision",
        "/World/Environment/GroundCollider",
    ]
    absent = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    visual_meshes = []
    collision_meshes = []
    point_prims = []
    transparent_values = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            colors = mesh.GetDisplayColorPrimvar()
            opacity = mesh.GetDisplayOpacityPrimvar()
            values = opacity.Get() if opacity and opacity.HasValue() else []
            if values:
                transparent_values.extend(float(value) for value in values if float(value) < 0.99)
            info = {
                "path": path,
                "points": len(mesh.GetPointsAttr().Get() or []),
                "faces": len(mesh.GetFaceVertexCountsAttr().Get() or []),
                "colors_present": bool(colors and colors.HasValue()),
                "collision_enabled": bool(
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                    else False
                ),
            }
            if path.startswith("/World/Environment/Visual"):
                visual_meshes.append(info)
            if path.startswith("/World/Environment/Collision"):
                collision_meshes.append(info)
        if prim.IsA(UsdGeom.Points) and path.startswith("/World/Environment/PointsAlternative"):
            points_geom = UsdGeom.Points(prim)
            point_colors = points_geom.GetDisplayColorPrimvar()
            point_prims.append(
                {
                    "path": path,
                    "points": len(points_geom.GetPointsAttr().Get() or []),
                    "colors_present": bool(point_colors and point_colors.HasValue()),
                }
            )
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Environment/Visual"))
    aligned = bound.ComputeAlignedRange()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    stage_bbox = {
        "min": [float(minimum[i]) for i in range(3)],
        "max": [float(maximum[i]) for i in range(3)],
        "dimensions": [float(maximum[i] - minimum[i]) for i in range(3)],
    }
    visual_visibility = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Environment/Visual")).GetVisibilityAttr().Get()
    points_visibility = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Environment/PointsAlternative")).GetVisibilityAttr().Get()
    collision_visibility = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Environment/Collision")).GetVisibilityAttr().Get()
    checks = {
        "all_references_exist": not missing,
        "no_absolute_asset_paths": not absolute,
        "required_hierarchy_present": not absent,
        "z_up": UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z,
        "meters_per_unit_is_1": UsdGeom.GetStageMetersPerUnit(stage) == 1.0,
        "visual_mesh_present": bool(visual_meshes),
        "visual_colors_present": bool(visual_meshes) and all(x["colors_present"] for x in visual_meshes),
        "visual_collision_disabled": bool(visual_meshes) and all(not x["collision_enabled"] for x in visual_meshes),
        "collision_mesh_present": bool(collision_meshes),
        "collision_enabled": bool(collision_meshes) and all(x["collision_enabled"] for x in collision_meshes),
        "collision_invisible": collision_visibility == UsdGeom.Tokens.invisible,
        "visual_opacity_opaque": not transparent_values,
        "point_alternative_present": bool(point_prims),
        "point_alternative_colors_present": bool(point_prims)
        and all(item["colors_present"] for item in point_prims),
        "default_visibility_matches_stage": (
            points_visibility != UsdGeom.Tokens.invisible if expect_points else visual_visibility != UsdGeom.Tokens.invisible
        ),
    }
    return {
        "stage": stage_path.name,
        "checks": checks,
        "passed": all(checks.values()),
        "references": resolved,
        "absolute_references": absolute,
        "missing_references": missing,
        "missing_required_prims": absent,
        "visual_meshes": visual_meshes,
        "collision_meshes": collision_meshes,
        "point_prims": point_prims,
        "point_count_total": sum(item["points"] for item in point_prims),
        "stage_visual_bbox": stage_bbox,
        "subunit_opacity_values": transparent_values[:20],
    }


def write_reports_and_readme(root: Path, processing: dict, validation: dict, point_validation: dict) -> None:
    visual = processing["visual_mesh"]
    collision = processing["collision_mesh"]
    clean = processing["clean"]
    raw = processing["raw"]
    origin = processing["coordinate_system"]["local_origin_source_coordinates"]
    report = f"""# Relatório de processamento

## Resultado

- Pontos originais: {raw['points_read']:,}
- XYZ inválidos: {raw['invalid_xyz']:,}
- Dimensões originais: {raw['bbox_utm']['dimensions'][0]:.3f} × {raw['bbox_utm']['dimensions'][1]:.3f} × {raw['bbox_utm']['dimensions'][2]:.3f} m
- Espaçamento NN mediano original (amostra): {raw['nearest_neighbour_m']['median']:.4f} m
- Nuvem limpa: {clean['points_after_outlier_filter']:,} pontos
- Outliers removidos: {clean['outliers_removed']:,} ({clean['outliers_removed_fraction']:.2%})
- Mesh visual: {visual['final']['vertices']:,} vértices, {visual['final']['triangles']:,} triângulos
- Mesh de colisão: {collision['final_vertices']:,} vértices, {collision['final_triangles']:,} triângulos
- Cores na mesh visual: {visual['colors_present']}
- Validação da stage principal: {'PASSOU' if validation['passed'] else 'FALHOU'}

## Coordenadas

O PLY tem coordenadas projetadas tipo UTM e está em metros, com Z vertical. Não foi aplicada rotação.
A geometria foi recentrada para precisão numérica. A origem local, em coordenadas da fonte, é:

`X={origin[0]:.9f}, Y={origin[1]:.9f}, Z={origin[2]:.9f}`

## Parâmetros efetivamente usados

- Limpeza: voxel 0,03 m; 16 vizinhos; `std_ratio=4,0`; normais num raio de 0,09 m.
- Visual: entrada a 0,05 m; Poisson depth 12; scale 1,05; corte de densidade 0,5%; suporte máximo 0,15 m; sem simplificação quadric da mesh visual.
- Colisão: entrada independente a 0,40 m; 10 vizinhos; `std_ratio=2,5`; Poisson depth 9; corte de densidade 2%; alvo 80 000 triângulos.
- Ball Pivoting foi testado com raios 0,055/0,09/0,15 m no piloto de 2 634 382 pontos e interrompido depois de mais de seis minutos sem terminar; Poisson foi escolhido por concluir e ter menor taxa medida de buracos.

## Validação

```json
{json.dumps(validation['checks'], indent=2, ensure_ascii=False)}
```
"""
    (root / "reports" / "report.md").write_text(report, encoding="utf-8")
    readme = f"""# Ambiente portátil para NVIDIA Isaac Sim

Abra no Isaac Sim:

`environment_ready.usda`

Este é o ficheiro principal: mesh visual RGB ativa, colisão invisível ativa e plano de segurança invisível por baixo do ponto mais baixo da nuvem limpa. Não é necessário executar scripts para o abrir.

## Escala e orientação

- A stage usa `metersPerUnit = 1` e `upAxis = Z`.
- A extensão visual validada é {validation['stage_visual_bbox']['dimensions'][0]:.3f} × {validation['stage_visual_bbox']['dimensions'][1]:.3f} × {validation['stage_visual_bbox']['dimensions'][2]:.3f} m.
- A origem UTM/local está guardada nos metadados de `/World/Environment` e no relatório: `{origin}`.
- Para confirmar no Isaac Sim, selecione `/World/Environment/Visual` e inspecione a bounding box; o chão deve ter Z próximo de {clean['bbox_local']['min'][2]:.3f} m.

## Colisões

- `/World/Environment/Visual`: `physics:collisionEnabled = false`.
- `/World/Environment/Collision`: invisível e com collider triangle mesh ativo.
- `/World/Environment/GroundCollider`: cubo/placa invisível de segurança abaixo da geometria.
- Para desativar colisões, desligue `Collision Enabled` nos prims de mesh sob `Collision` e no `GroundCollider`, ou desative esses prims.

## Alternativa visual em pontos

`environment_points_ready.usda` ativa a representação em pontos/splats e oculta a mesh. É útil para comparar vegetação, postes e detalhes finos que Poisson pode suavizar ou unir. No ficheiro principal, a mesma alternativa existe em `/World/Environment/PointsAlternative`, invisível por omissão.

## Transportar para outro computador

Copie toda a pasta de saída sem alterar a estrutura interna. O USDA contém apenas referências relativas.

## Repetir o pipeline

Use o wrapper; ele processa a nuvem, cria os chunks de pontos, converte os assets e valida a stage:

```bash
scripts/run_pipeline.sh INPUT.ply NOVA_PASTA /caminho/para/isaac-sim
```

Os valores usados estão em `reports/processing_report.json`; ajuste-os apenas depois de medir a densidade do novo PLY. Para nuvens com outro referencial, confirme primeiro metros e Z-up.

## Conteúdo principal

- `visual/visual_mesh.usd`: asset visual RGB usado pela stage.
- `visual/visual_mesh.ply`, `.obj`, `.glb`: intermediários portáteis.
- `visual/visual_points.usda`: agregador da alternativa em 836 325 pontos RGB (voxel 0,10 m, largura 0,09 m).
- `collision/collision_mesh.usd`: collider simplificado.
- `pointcloud/map_clean.ply`: nuvem limpa com RGB e normais.
- `reports/report.md`: relatório legível.
- `reports/validation_report.json`: verificações automáticas.
- `reports/previews/`: pré-visualizações.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def apply_optional_crop_metrics(root: Path, processing: dict) -> dict:
    """Use cutaway metrics for the final stage when crop_environment.py was run."""
    path = root / "reports" / "interior_refinement_report.json"
    if not path.exists():
        return processing
    crop = json.loads(path.read_text(encoding="utf-8"))
    processing["clean"]["bbox_local"] = crop["interior_cloud_bbox"]
    processing["clean"]["points_after_outlier_filter"] = crop["interior_clean_points"]
    processing["visual_mesh"]["final"] = crop["visual_mesh"]["after"]
    processing["visual_mesh"]["colors_present"] = True
    collision_after = crop["collision_mesh"]["after"]
    processing["collision_mesh"]["final_vertices"] = collision_after["vertices"]
    processing["collision_mesh"]["final_triangles"] = collision_after["triangles"]
    processing["collision_mesh"]["bbox_local"] = collision_after["bbox_local"]
    processing["final_crop"] = {
        "bounds_local_m": crop["bounds_local_m"],
        "floor_z_m": crop["floor_z_m"],
    }
    return processing


def write_portable_reports(root: Path, processing: dict, validation: dict) -> None:
    visual = processing["visual_mesh"]
    collision = processing["collision_mesh"]
    clean = processing["clean"]
    raw = processing["raw"]
    params = processing["parameters"]
    origin = processing["coordinate_system"]["local_origin_source_coordinates"]
    dims = validation["stage_visual_bbox"]["dimensions"]
    report = f"""# Processing report

- Source points: {raw['points_read']:,}
- Invalid XYZ: {raw['invalid_xyz']:,}
- Source dimensions: {raw['bbox_utm']['dimensions'][0]:.3f} × {raw['bbox_utm']['dimensions'][1]:.3f} × {raw['bbox_utm']['dimensions'][2]:.3f} m
- Measured median nearest-neighbour spacing: {raw['nearest_neighbour_m']['median']:.6f} m
- Clean points: {clean['points_after_outlier_filter']:,}
- Visual mesh: {visual['final']['vertices']:,} vertices, {visual['final']['triangles']:,} triangles
- Collision mesh: {collision['final_vertices']:,} vertices, {collision['final_triangles']:,} triangles
- Vertex colours: {visual['colors_present']}
- Final validation: {'PASSED' if validation['passed'] else 'FAILED'}

## Coordinate handling

The source is treated as metres and Z-up. Geometry is recentered for numerical
precision. The source-coordinate origin stored as USD metadata is:

`X={origin[0]:.9f}, Y={origin[1]:.9f}, Z={origin[2]:.9f}`

## Effective parameters

- Clean voxel: {params['clean_voxel']:.6f} m
- Visual voxel: {params['visual_voxel']:.6f} m
- Visual Poisson depth: {params['visual_depth']}
- Collision voxel: {params['collision_voxel']:.6f} m
- Collision Poisson depth: {params['collision_depth']}
- Collision target: {params['collision_triangles']:,} triangles
- Point alternative voxel: {params['points_voxel']:.6f} m

## Validation checks

```json
{json.dumps(validation['checks'], indent=2)}
```
"""
    (root / "reports" / "report.md").write_text(report, encoding="utf-8")
    readme = f"""# Portable NVIDIA Isaac Sim environment

Open **`environment_ready.usda`** in Isaac Sim. It contains the coloured
visual mesh, an invisible independent collision mesh, and an invisible safety
ground collider.

The stage is Z-up, uses metres (`metersPerUnit = 1`), and its validated visual
dimensions are {dims[0]:.3f} × {dims[1]:.3f} × {dims[2]:.3f} m.

To inspect the point representation instead of the reconstructed surface, open
`environment_points_ready.usda`.

## Collision

- `/World/Environment/Visual`: collision disabled.
- `/World/Environment/Collision`: invisible, collision enabled.
- `/World/Environment/GroundCollider`: invisible safety plane.

## Moving to another computer

Copy this entire output directory without changing its internal structure.
All USD references are relative; the original PLY is not required to open the
finished environment.

See `reports/report.md` and `reports/processing_report.json` for parameters
and measurements.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--point-width", type=float, default=0.09)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-points", action="store_true")
    args = parser.parse_args()
    if not args.skip_convert or not args.skip_points:
        raise ValueError(
            "This portable finalizer expects USD assets authored by geometry_to_usd.py; "
            "pass --skip-convert --skip-points."
        )
    root = args.root.expanduser().resolve()
    copy_self(root)
    processing_path = root / "reports" / "processing_report.json"
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    processing = apply_optional_crop_metrics(root, processing)

    app = None
    try:
        visual_glb = root / "visual" / "visual_mesh.glb"
        collision_glb = root / "collision" / "collision_mesh.glb"
        visual_usd = root / "visual" / "visual_mesh.usd"
        collision_usd = root / "collision" / "collision_mesh.usd"
        if not args.skip_convert:
            enable_extension("omni.kit.asset_converter")
            loop = asyncio.get_event_loop()
            log("Converting visual GLB to USD")
            loop.run_until_complete(convert_asset(visual_glb, visual_usd))
            log("Converting collision GLB to USD")
            loop.run_until_complete(convert_asset(collision_glb, collision_usd))
        elif not visual_usd.exists() or not collision_usd.exists():
            raise FileNotFoundError("--skip-convert requested but one or more mesh USD assets are missing")
        if not args.skip_points:
            points_npz = root / "visual" / "visual_points_splats.npz"
            if not points_npz.exists():
                points_npz = root / "visual" / "visual_points.npz"
            points_stats = create_points_usd(
                points_npz,
                root / "visual" / "visual_points.usd",
                args.point_width,
            )
        else:
            points_stats = {"reused_existing_usd": True, "width_m": args.point_width}
        points_asset = root / "visual" / "visual_points.usda"
        if not points_asset.exists():
            points_asset = root / "visual" / "visual_points.usd"
        assets = {
            "visual": inspect_asset(visual_usd),
            "collision": inspect_asset(collision_usd),
            "points": points_stats,
        }
        (root / "reports" / "usd_assets.json").write_text(
            json.dumps(assets, indent=2), encoding="utf-8"
        )

        log("Assembling environment_ready.usda")
        primary_path = root / "environment_ready.usda"
        primary_build = create_stage(
            primary_path,
            visual_usd,
            points_asset,
            collision_usd,
            processing,
            show_points=False,
        )
        log("Assembling environment_points_ready.usda")
        point_path = root / "environment_points_ready.usda"
        point_build = create_stage(
            point_path,
            visual_usd,
            points_asset,
            collision_usd,
            processing,
            show_points=True,
        )
        validation = validate_stage(primary_path, expect_points=False)
        point_validation = validate_stage(point_path, expect_points=True)
        validation["build"] = primary_build
        point_validation["build"] = point_build
        (root / "reports" / "validation_report.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        (root / "reports" / "validation_points_report.json").write_text(
            json.dumps(point_validation, indent=2), encoding="utf-8"
        )
        write_portable_reports(root, processing, validation)
        log(f"Primary validation passed: {validation['passed']}")
        log(f"Point alternative validation passed: {point_validation['passed']}")
        if not validation["passed"] or not point_validation["passed"]:
            return 3
        log(f"Final stage: {primary_path}")
        return 0
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
import open3d as o3d


COLOR_NAMES = {
    "red",
    "green",
    "blue",
    "alpha",
    "r",
    "g",
    "b",
    "diffuse_red",
    "diffuse_green",
    "diffuse_blue",
}


def read_ply_header(path: Path, max_lines: int = 256) -> list[str]:
    lines: list[str] = []
    with path.open("rb") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            text = line.decode("ascii", errors="ignore").strip()
            lines.append(text)
            if text == "end_header":
                break
    return lines


def parse_vertex_properties(header_lines: list[str]) -> tuple[str | None, int | None, list[str]]:
    ply_format = None
    vertex_count = None
    vertex_properties: list[str] = []
    inside_vertex = False

    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            ply_format = parts[1]
        elif parts[:2] == ["element", "vertex"] and len(parts) >= 3:
            vertex_count = int(parts[2])
            inside_vertex = True
        elif parts[0] == "element" and parts[1] != "vertex":
            inside_vertex = False
        elif inside_vertex and parts[0] == "property":
            vertex_properties.append(parts[-1])

    return ply_format, vertex_count, vertex_properties


def nn_stats(pcd: o3d.geometry.PointCloud, sample_size: int) -> dict[str, float] | None:
    points = np.asarray(pcd.points)
    if len(points) < 2:
        return None

    rng = np.random.default_rng(0)
    take = min(sample_size, len(points))
    sample_idx = rng.choice(len(points), size=take, replace=False)

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for idx in sample_idx:
        _, _, dist2 = kdtree.search_knn_vector_3d(pcd.points[idx], 2)
        if len(dist2) > 1:
            dists.append(math.sqrt(dist2[1]))

    if not dists:
        return None

    arr = np.asarray(dists)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def estimate_dominant_plane(
    pcd: o3d.geometry.PointCloud, median_nn: float | None
) -> tuple[np.ndarray, float, int] | None:
    points = np.asarray(pcd.points)
    if len(points) < 5000:
        return None

    extent = points.max(axis=0) - points.min(axis=0)
    base_voxel = max(float(np.max(extent)) / 500.0, 0.02)
    if median_nn is not None:
        base_voxel = max(base_voxel, median_nn * 3.0)

    sample = pcd.voxel_down_sample(base_voxel)
    if len(sample.points) < 1000:
        sample = pcd

    distance_threshold = max(base_voxel * 0.75, 0.02)
    if median_nn is not None:
        distance_threshold = max(distance_threshold, median_nn * 2.0)

    plane_model, inliers = sample.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=500,
    )
    normal = np.asarray(plane_model[:3], dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    normal /= norm
    inlier_ratio = len(inliers) / max(len(sample.points), 1)
    return normal, inlier_ratio, len(inliers)


def axis_name_from_normal(normal: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(normal)))
    return ["X", "Y", "Z"][axis]


def scale_hint(extent: np.ndarray, median_nn: float | None) -> str:
    max_extent = float(np.max(extent))
    if max_extent > 1000:
        return "Escala suspeita: o mapa e muito grande. Confirma se esta em milimetros ou centimetros."
    if max_extent < 1.0:
        return "Escala suspeita: o mapa e muito pequeno. Confirma se nao esta em centimetros ou milimetros."
    if median_nn is not None and median_nn > 1.0:
        return "Espacamento medio alto para um mapa em metros. Confirma se a unidade nao esta em centimetros."
    return "Nao ha alerta obvio de escala, mas confirma sempre contra uma medida conhecida."


def density_hint(num_points: int, median_nn: float | None) -> str:
    if num_points > 20_000_000:
        return "Muito pesada para simulacao direta. Downsampling quase obrigatorio."
    if num_points > 5_000_000:
        return "Pesada. Recomenda-se voxel downsampling antes de qualquer mesh."
    if median_nn is not None and median_nn > 0.25:
        return "Nuvem relativamente esparsa para reconstruir superficies finas."
    return "Densidade parece utilizavel, dependendo do tipo de ambiente e do metodo de meshing."


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspeciona um ficheiro PLY para uso em Isaac Sim.")
    parser.add_argument("input", type=Path, help="Caminho para o .ply")
    parser.add_argument(
        "--nn-sample-size",
        type=int,
        default=10000,
        help="Numero de pontos usados para estimar distancia ao vizinho mais proximo.",
    )
    args = parser.parse_args()

    header_lines = read_ply_header(args.input)
    ply_format, vertex_count, vertex_properties = parse_vertex_properties(header_lines)
    pcd = o3d.io.read_point_cloud(str(args.input))

    if len(pcd.points) == 0:
        print(f"[ERRO] Nao foi possivel ler pontos de {args.input}")
        return 2

    points = np.asarray(pcd.points)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    center = points.mean(axis=0)
    nn = nn_stats(pcd, args.nn_sample_size)
    plane = estimate_dominant_plane(pcd, nn["median"] if nn else None)

    properties = set(vertex_properties)
    has_xyz = {"x", "y", "z"}.issubset(properties) or len(points) > 0
    has_rgb = bool(COLOR_NAMES.intersection(properties)) or pcd.has_colors()

    print(f"Ficheiro: {args.input}")
    print(f"Formato PLY: {ply_format or 'desconhecido'}")
    print(f"Vertices no header: {vertex_count if vertex_count is not None else 'desconhecido'}")
    print(f"Properties no vertex: {', '.join(vertex_properties) if vertex_properties else 'nenhuma'}")
    print()
    print("Validacao base")
    print(f"- XYZ: {'sim' if has_xyz else 'nao'}")
    print(f"- RGB: {'sim' if has_rgb else 'nao'}")
    print(f"- Normais: {'sim' if pcd.has_normals() else 'nao'}")
    print(f"- Numero de pontos: {len(points)}")
    print()
    print("Bounding box")
    print(f"- Min: {bbox_min}")
    print(f"- Max: {bbox_max}")
    print(f"- Extent: {extent}")
    print(f"- Centro medio: {center}")
    print()
    print("Escala e densidade")
    print(f"- Escala: {scale_hint(extent, nn['median'] if nn else None)}")
    if nn is not None:
        print(f"- Distancia NN media: {nn['mean']:.4f}")
        print(f"- Distancia NN mediana: {nn['median']:.4f}")
        print(f"- Distancia NN p90: {nn['p90']:.4f}")
        print(
            "- Voxel sugerido: "
            f"{nn['median'] * 1.5:.3f} fino, "
            f"{nn['median'] * 3.0:.3f} medio, "
            f"{nn['median'] * 6.0:.3f} agressivo"
        )
    print(f"- Densidade: {density_hint(len(points), nn['median'] if nn else None)}")
    print()

    if plane is not None:
        normal, inlier_ratio, inlier_count = plane
        print("Orientacao")
        print(f"- Normal do maior plano estimado: {normal}")
        print(f"- Eixo mais alinhado: {axis_name_from_normal(normal)}")
        print(f"- Inliers do maior plano: {inlier_count} ({inlier_ratio:.1%} da amostra usada)")
        print(
            "- Interpretacao: se esse plano for o chao, o eixo acima deve ser o 'up' do mapa. "
            "No Isaac Sim o mundo e Z-up."
        )
    else:
        print("Orientacao")
        print("- Nao foi possivel estimar um plano dominante com confianca.")

    if not has_rgb:
        print()
        print("Aviso: este PLY nao parece conter RGB. Para cor no Isaac Sim vais precisar de outro PLY colorido,")
        print("ou entao trabalhar sem cor, com materiais simples, ou reprojectar as imagens novamente.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# utils.py
import pandas as pd
import numpy as np
import math
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

# ---------------------- 기존 함수 ----------------------
def find_closest_node_id(target_point, node_df):
    min_dist = float('inf')
    closest_node_id = None
    for _, row in node_df.iterrows():
        node_coords = (row['경도'], row['위도'])
        dist = math.sqrt((target_point[0] - node_coords[0])**2 + (target_point[1] - node_coords[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_node_id = row['NODE_ID']
    return closest_node_id

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def filter_nodes_by_radius(node_df, difficult_segments, radius_km=2.0):
    if not difficult_segments:
        return node_df.copy()
    filtered_indices = []
    node_lat = node_df['위도'].values
    node_lon = node_df['경도'].values
    for segment in difficult_segments:
        start_point = segment['start_point']
        end_point = segment['end_point']
        dist_to_start = haversine(node_lat, node_lon, start_point[1], start_point[0])
        dist_to_end = haversine(node_lat, node_lon, end_point[1], end_point[0])
        in_radius_indices = np.where((dist_to_start <= radius_km) | (dist_to_end <= radius_km))[0]
        filtered_indices.extend(node_df.iloc[in_radius_indices].index.tolist())
    return node_df.loc[list(set(filtered_indices))]

def replace_segment(original_path_points, start_point, end_point, easy_path_points):
    start_idx = original_path_points.index(start_point)
    end_idx = original_path_points.index(end_point) + 1
    new_path = original_path_points[:start_idx] + easy_path_points + original_path_points[end_idx:]
    return new_path

def build_edge_link_maps(
    link_df,
    u_col='시작 노드', v_col='끝 노드', id_col='LINK_ID'
):
    """
    링크 CSV에서 (u,v)->[link_id...] (방향성), frozenset({u,v})->[link_id...] (무방향) 맵 생성
    """
    edge_dir = {}
    edge_undir = {}
    for _, row in link_df.iterrows():
        try:
            u = int(row[u_col]); v = int(row[v_col]); lid = int(row[id_col])
        except Exception:
            continue
        edge_dir.setdefault((u, v), []).append(lid)
        edge_undir.setdefault(frozenset((u, v)), []).append(lid)
    return edge_dir, edge_undir

def nodes_to_link_ids(
    path_nodes: List[int],
    edge_dir_map: Dict[Tuple[int,int], List[int]],
    edge_undir_map: Dict[frozenset, List[int]]
) -> List[int]:
    """
    노드 경로 → 링크 ID 시퀀스. (u,v)우선, 없으면 (v,u), 그래도 없으면 무방향 매핑.
    여러 개면 첫 번째 사용.
    """
    link_ids = []
    for i in range(len(path_nodes)-1):
        u = path_nodes[i]; v = path_nodes[i+1]
        lids = edge_dir_map.get((u, v)) or edge_dir_map.get((v, u)) or edge_undir_map.get(frozenset((u, v)))
        if lids:
            link_ids.append(int(lids[0]))
    return link_ids
# ---------------------- 추가: 라우팅 계산용 유틸 ----------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def edge_length_m(u: int, v: int, node_xy: Dict[int, Tuple[float, float]]) -> float:
    lon1, lat1 = node_xy[u]
    lon2, lat2 = node_xy[v]
    return haversine_m(lat1, lon1, lat2, lon2)

def ensure_edge_metrics(graph: nx.Graph, node_xy: Dict[int, Tuple[float, float]]):
    for u, v, data in graph.edges(data=True):
        if 'length_m' not in data:
            data['length_m'] = edge_length_m(u, v, node_xy)
        link_diff = float(data.get('weight', 0.0))
        data['eff_weight'] = max(data['length_m'], 1.0) * link_diff

def normalize_edge_weights(graph: nx.Graph, node_xy: Dict[int, Tuple[float, float]], method: str = "p95"):
    vals = []
    for _, _, data in graph.edges(data=True):
        w = data.get("weight", None)
        if w is not None and np.isfinite(w):
            vals.append(float(w))
    if not vals:
        return
    p95 = np.percentile(vals, 95)
    if not np.isfinite(p95) or p95 <= 0:
        p95 = max(vals) or 1.0
    for u, v, data in graph.edges(data=True):
        raw = float(data.get("weight", 0.0))
        data["weight_raw"] = raw
        norm = raw / p95
        norm = 0.0 if norm < 0.0 else (1.0 if norm > 1.0 else norm)
        data["weight"] = norm
        if "length_m" not in data:
            data["length_m"] = edge_length_m(u, v, node_xy)
        data["eff_weight"] = max(data["length_m"], 1.0) * data["weight"]

def path_avg_difficulty(graph: nx.Graph,
                        path_nodes: List[int],
                        node_xy: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float]:
    if not path_nodes or len(path_nodes) < 2:
        return float('inf'), 0.0, float('inf')
    total_len = 0.0
    weighted_sum = 0.0
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        if u not in node_xy or v not in node_xy:
            return float('inf'), 0.0, float('inf')
        data = graph.get_edge_data(u, v) or {}
        length_m = data.get('length_m')
        if length_m is None:
            length_m = edge_length_m(u, v, node_xy)
        link_diff = float(data.get('weight', 0.0))
        total_len += length_m
        weighted_sum += link_diff * length_m
    if total_len <= 0:
        return float('inf'), 0.0, float('inf')
    avg_diff = weighted_sum / total_len
    return avg_diff, total_len, weighted_sum

def segment_length_m(start_point: List[float], end_point: List[float]) -> float:
    sx, sy = start_point
    ex, ey = end_point
    return haversine_m(sy, sx, ey, ex)

def build_path_points_from_nodes(path_nodes: List[int],
                                node_xy: Dict[int, Tuple[float, float]]) -> List[List[float]]:
    pts = []
    for n in path_nodes:
        if n in node_xy:
            x, y = node_xy[n]
            pts.append([float(x), float(y)])
    return pts

def k_alt_paths_between(graph: nx.Graph, source: int, target: int, k: int = 5) -> List[List[int]]:
    try:
        gen = nx.shortest_simple_paths(graph, source, target, weight="eff_weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    paths = []
    for i, p in enumerate(gen):
        if i >= k:
            break
        paths.append(p)
    return paths

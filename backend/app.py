# app.py
import os, json, requests
import pandas as pd
import numpy as np
import networkx as nx
import math, time, logging
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import uvicorn
from dataclasses import dataclass
from typing import Optional
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum  # 이미 import 되어 있으면 생략
# === 성능 패치: KD-Tree & 캐시 ===
from scipy.spatial import cKDTree
from functools import lru_cache

from dataclasses import dataclass
import statistics

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("navi")

# --- 외부 모듈 ---
from difficulty_scorer import DifficultyScorer, InitialWeights
from graph_builder import create_graph_from_csv
from utils import find_closest_node_id, filter_nodes_by_radius, replace_segment
# 팀원 코드 (차선 이벤트)
from cal_lane_chng import LaneChangeCalculator

# ===================== 전역 =====================
G: nx.Graph = nx.Graph()
node_coords_map: Dict[int, Tuple[float, float]] = {}
difficulty_scorer: DifficultyScorer
diff_node: pd.DataFrame
diff_link: pd.DataFrame
lane_calc: LaneChangeCalculator

# 링크 매핑
edge_dir_map: Dict[Tuple[int, int], int] = {}
edge_undir_map: Dict[Tuple[int, int], int] = {}

# 공공 속도 테이블(선택, 훅)
PUBLIC_SPEEDS: Dict[int, float] = {}  # link_id -> speed(m/s) 등, 없으면 훅에서 기본값 사용

# 파라미터/가드
MAX_LENGTH_INCREASE_RATIO_LOCAL = 2.0       # 우회 길이 증가율 가드(세그먼트)
K_SEGMENT = 3                                # ★패치: 세그먼트별 후보 개수(종전 5→3, 과탐색 완화)
SEGMENT_DIFFICULTY_THRESHOLD = 0.95          # 위험구간 임계값
W_LANE = 0.20                                # 차선 벌점 가중치(경로/세그먼트 공통)

# === 성능 패치: KD-Tree 전역 객체 ===
node_kdtree: Optional[cKDTree] = None
node_id_array: Optional[np.ndarray] = None
node_xy_array: Optional[np.ndarray] = None

# ===================== 유틸/헬퍼 =====================
@dataclass
class SamplingParams:
    max_points: int = 30
    epsilon_base: float = 25.0
    alpha: float = 1.2
    beta: float = 0.7
    theta_turn: float = 25.0
    theta_uturn: float = 135.0
    d_min: float = 80.0
    d_max: float = 800.0
    # --- 호환용 별칭(선택) ---
    epsilon_base_m: Optional[float] = None
    d_min_m: Optional[float] = None
    d_max_m: Optional[float] = None

    def __post_init__(self):
        if self.epsilon_base_m is not None:
            self.epsilon_base = float(self.epsilon_base_m)
        if self.d_min_m is not None:
            self.d_min = float(self.d_min_m)
        if self.d_max_m is not None:
            self.d_max = float(self.d_max_m)


@dataclass
class SamplingWeights:
    w_turn: float = 0.35
    w_uturn: float = 0.25
    w_lane: float = 0.20
    w_intersection: float = 0.10
    w_class: float = 0.05
    w_hazard: float = 0.05

def _angle_deg(a, b, c):
    # a,b,c: [lon,lat]
    # 벡터 BA, BC 사이 각도 (deg, 0~180)
    import math
    bax = a[0] - b[0]; bay = a[1] - b[1]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]
    na = math.hypot(bax, bay); nb = math.hypot(bcx, bcy)
    if na == 0 or nb == 0: return 0.0
    cosv = max(-1.0, min(1.0, (bax*bcx + bay*bcy) / (na*nb)))
    return math.degrees(math.acos(cosv))

def _segment_len_m(p, q):
    # p,q: [lon,lat] -> meters
    return haversine_m(p[1], p[0], q[1], q[0])

def _perp_dist_m(p, a, b):
    # P에서 AB 세그먼트까지 수선거리(m)
    # 거리/투영 계산은 구면 정밀도까지 필요치 않으므로 근사(Equirectangular)로 충분.
    ax, ay = a[0], a[1]; bx, by = b[0], b[1]; px, py = p[0], p[1]
    if ax == bx and ay == by:
        return _segment_len_m(p, a)
    # 투영 t
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    t = max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
    proj = [ax + t*vx, ay + t*vy]
    return _segment_len_m(p, proj)

def _extract_features(points, theta_turn=25.0, theta_uturn=135.0):
    n = len(points)
    turns = [0.0]*n
    is_uturn = [0]*n
    # 간단한 교차로/분기 후보: 회전각이 크거나, 곡률이 급변하는 지점
    is_intersection = [0]*n
    # 차선변경 후보(메타 없을 때): 회전각 중간(15~60) + 연속 두 세그 구간 길이 짧음
    lane_change = [0]*n
    class_change = [0]*n
    hazard = [0.0]*n

    # 길이정보
    seg_len = [0.0]*(n-1)
    for i in range(n-1):
        seg_len[i] = _segment_len_m(points[i], points[i+1])

    for i in range(1, n-1):
        ang = _angle_deg(points[i-1], points[i], points[i+1])
        turns[i] = ang
        if ang >= theta_uturn: is_uturn[i] = 1
        if ang >= theta_turn: is_intersection[i] = 1

        # 간이 lane-change 후보: 중간 회전각 + 짧은 연속 세그
        if 15 <= ang <= 60:
            left = seg_len[i-1] if i-1 < len(seg_len) else 0
            right = seg_len[i] if i < len(seg_len) else 0
            if left < 80 or right < 80:
                lane_change[i] = 1

    return {
        "turns": turns,
        "is_uturn": is_uturn,
        "is_intersection": is_intersection,
        "lane_change": lane_change,
        "class_change": class_change,
        "hazard": hazard,
        "seg_len": seg_len,
    }

def _score_importance(feat, weights: SamplingWeights, theta_turn_clip=90.0):
    import math
    turns = feat["turns"]
    n = len(turns)
    imp = [0.0]*n
    for i in range(n):
        f_turn = min(1.0, abs(turns[i]) / theta_turn_clip)
        imp[i] = (
            weights.w_turn * f_turn +
            weights.w_uturn * feat["is_uturn"][i] +
            weights.w_lane * feat["lane_change"][i] +
            weights.w_intersection * feat["is_intersection"][i] +
            weights.w_class * feat["class_change"][i] +
            weights.w_hazard * feat["hazard"][i]
        )
    return imp

def _required_indices(n, feat, imp, tau=0.6):
    req = set()
    if n > 0: req.add(0)
    if n > 1: req.add(n-1)
    for i in range(n):
        if feat["is_uturn"][i] or feat["is_intersection"][i] or feat["lane_change"][i] or imp[i] >= tau:
            req.add(i)
    return req

def _weighted_rdp(points, imp, params: SamplingParams, keep:set):
    # RDP with per-point epsilon: eps_i = eps_base / (1 + alpha*imp[i])
    n = len(points)
    if n <= 2:
        return list(range(n)), []  # kept, removed

    kept = set(keep)  # 필수 보존
    removed = set()

    def rdp(l, r):
        # [l, r] 구간
        a = points[l]; b = points[r]
        max_d = -1.0; idx = -1
        for i in range(l+1, r):
            eps_i = params.epsilon_base / (1.0 + params.alpha*imp[i])
            d = _perp_dist_m(points[i], a, b) - eps_i
            if d > max_d:
                max_d = d; idx = i
        if max_d > 0:  # 허용오차 초과 → 분기
            rdp(l, idx)
            rdp(idx, r)
        else:
            # 구간 내 비필수점 제거
            for i in range(l+1, r):
                if i not in keep:
                    removed.add(i)
            kept.add(l); kept.add(r)

    rdp(0, n-1)
    kept_idx = sorted(list(kept))
    removed_idx = sorted(list(removed - kept))
    return kept_idx, removed_idx

def _adaptive_pacing(points, kept_idx, imp, params: SamplingParams):
    # 남는 슬롯을 구간별 중요도/거리 기반으로 분배
    n = len(points)
    if n == 0: return []
    kept_idx = sorted(set(kept_idx))
    if kept_idx[0] != 0: kept_idx = [0] + kept_idx
    if kept_idx[-1] != n-1: kept_idx = kept_idx + [n-1]

    # 현재 보유 수
    cur = len(kept_idx)
    if cur >= params.max_points:
        return kept_idx[:params.max_points]

    # 구간 통계
    segs = []
    for a, b in zip(kept_idx[:-1], kept_idx[1:]):
        # 거리
        d = 0.0
        for i in range(a, b):
            d += _segment_len_m(points[i], points[i+1])
        # 중요도 합
        g = sum(imp[a:b+1])
        segs.append({"a":a, "b":b, "D":d, "G":g})

    # 정규화 가중
    sumD = sum(s["D"] for s in segs) or 1.0
    sumG = sum(s["G"] for s in segs) or 1.0
    for s in segs:
        pD = s["D"]/sumD
        pG = s["G"]/sumG
        s["P"] = params.beta*pG + (1-params.beta)*pD

    L = params.max_points - cur
    # 각 구간별 추가 개수
    alloc = [max(0, round(L * s["P"])) for s in segs]
    for i, s in enumerate(segs):
        # 이 구간 길이로 넣을 수 있는 이론적 최대(최소간격 d_min_m 기준)
        max_for_seg = max(0, int(s["D"] // (params.d_min_m * 1.1)))
        if alloc[i] > max_for_seg:
            alloc[i] = max_for_seg
    # 합이 L과 다를 수 있음 → 보정
    diff = L - sum(alloc)
    # diff>0 이면 P 큰 구간부터 +1, diff<0 이면 P 작은 구간부터 -1
    order = sorted(range(len(segs)), key=lambda i: segs[i]["P"], reverse=(diff>0))
    j = 0
    while diff != 0 and j < len(order):
        i = order[j]; j += 1
        if diff > 0:
            alloc[i] += 1; diff -= 1
        else:
            if alloc[i] > 0:
                alloc[i] -= 1; diff += 1

    # 구간별로 균등 삽입(간격 d_min 보장)
    new_idx = set(kept_idx)
    for s, cnt in zip(segs, alloc):
        if cnt <= 0: continue
        a, b = s["a"], s["b"]
        seg_len_m = s["D"]
        if seg_len_m <= 0: continue
        # a~b 사이에 cnt개 배치: 누적 거리 기준 균등
        target_ds = [(k+1)/(cnt+1) * seg_len_m for k in range(cnt)]
        # 실제 누적거리 따라 가장 가까운 인덱스를 선택(중복 방지)
        cum = 0.0
        t_idx = 0
        for i in range(a, b):
            edge = _segment_len_m(points[i], points[i+1])
            while t_idx < len(target_ds) and cum + edge >= target_ds[t_idx]:
                # 이 지점에 삽입 후보: i 또는 i+1 중 더 멀리 떨어진 쪽
                cand = i+1
                if cand not in new_idx:
                    new_idx.add(cand)
                t_idx += 1
            cum += edge

    final_idx = sorted(list(new_idx))
    # d_min 보장: 너무 가까운 점 제거(중요도 낮은 순)
    pruned = []
    last = None
    for idx in final_idx:
        if last is None:
            pruned.append(idx); last = idx
        else:
            d = _segment_len_m(points[last], points[idx])
            if d >= params.d_min_m:
                pruned.append(idx); last = idx
            else:
                # 가까우면 중요도 작은 쪽 제거
                if imp[idx] > imp[last]:
                    pruned[-1] = idx
                    last = idx
                # else: idx 버림
    return pruned[:params.max_points]

def _rebalance_to_limit(idx_list, imp, max_points):
    idx = sorted(idx_list)
    if len(idx) <= max_points:
        return idx, []
    # 중요도 낮고 회전각 작은 순으로 제거
    removed = []
    # 간단히 imp 기준 정렬해서 뒤에서 제거
    removable = idx[1:-1]  # 시작/끝 보호
    removable_sorted = sorted(removable, key=lambda i: imp[i])
    need = len(idx) - max_points
    to_remove = set(removable_sorted[:need])
    out = [i for i in idx if i not in to_remove]
    removed = sorted(list(to_remove))
    return out, removed

def _validate_fill(points, idx, params: SamplingParams):
    # d_max 초과 구간이 있으면 중간점을 더 넣고, 초과면 그대로(30 제한 우선)
    out = list(idx)
    changed = False
    i = 0
    while i < len(out)-1 and len(out) < params.max_points:
        a, b = out[i], out[i+1]
        d = _segment_len_m(points[a], points[b])
        if d > params.d_max_m:
            mid = (a + b)//2
            if mid not in out and mid > a and mid < b:
                out.insert(i+1, mid); changed = True
            else:
                i += 1
        else:
            i += 1
    return out, changed

def build_easy_path_waypoints(path_points: List[List[float]],
                            params: SamplingParams = SamplingParams(),
                            weights: SamplingWeights = SamplingWeights()):
    """
    입력: path_points = [[lon,lat], ...] (EASY 최종 경로)
    출력: {path_point_30, kept_required, removed_indices, score_stats, keyword, metrics...}
    """
    n = len(path_points)
    feat = _extract_features(path_points, params.theta_turn, params.theta_uturn)
    imp = _score_importance(feat, weights)

    # 필수점 + WRDP
    req = _required_indices(n, feat, imp, tau=0.6)
    kept_rdp, removed_rdp = _weighted_rdp(path_points, imp, params, keep=req)

    # 적응형 페이싱으로 슬롯 채우기
    idx = _adaptive_pacing(path_points, kept_rdp, imp, params)

    # 리밸런싱(혹시 초과 시)
    idx, removed_over = _rebalance_to_limit(idx, imp, params.max_points)

    # d_max 검증/보정(30 제한 범위 내에서만)
    idx, _ = _validate_fill(path_points, idx, params)

    # 최종 좌표
    path_30 = [[float(path_points[i][1]), float(path_points[i][0])] for i in idx]  # [lat,lng]로 반환

    # 간단 메트릭
    total_dist = 0.0
    for i in range(n-1):
        total_dist += _segment_len_m(path_points[i], path_points[i+1])

    # lane change / uturn 추정치
    lane_changes = sum(feat["lane_change"])
    u_turns = sum(feat["is_uturn"])

    # 중요도 통계
    vals = [imp[i] for i in idx]
    score_stats = {
        "min": round(min(vals), 4) if vals else 0.0,
        "p50": round(statistics.median(vals), 4) if vals else 0.0,
        "p90": round(np.percentile(vals, 90), 4) if vals else 0.0,
        "max": round(max(vals), 4) if vals else 0.0,
    }

    # 키워드
    keyword = "안전경로"
    if lane_changes <= 1: keyword += "|차선변경적음"
    if u_turns == 0: keyword += "|유턴없음"

    return {
        "path_point_raw_count": n,
        "path_point_30": path_30,  # [lat,lng] 배열 (카카오 경유지 입력에 바로 사용)
        "keyword": keyword,
        "duration": None,           # (원하면 kakao summary에서 채워넣기)
        "distance": round(total_dist, 1),
        "lane_changes": int(lane_changes),
        "u_turns": int(u_turns),
        "debug": {
            "kept_required": sorted(list(req)),
            "removed_indices": sorted(list(set(removed_rdp + removed_over))),
            "score_stats": score_stats
        }
    }

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
        if norm < 0.0: norm = 0.0
        if norm > 1.0: norm = 1.0
        data["weight"] = norm
        if "length_m" not in data:
            data["length_m"] = edge_length_m(u, v, node_xy)
        data["eff_weight"] = max(data["length_m"], 1.0) * data["weight"]

def path_avg_difficulty(graph: nx.Graph, path_nodes: List[int], node_xy: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float]:
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

def build_path_points_from_nodes(path_nodes: List[int], node_xy: Dict[int, Tuple[float, float]]) -> List[List[float]]:
    pts = []
    for n in path_nodes:
        if n in node_xy:
            x, y = node_xy[n]
            pts.append([float(x), float(y)])
    return pts

def pairwise_nodes(nodes: List[int]):
    for i in range(len(nodes) - 1):
        yield nodes[i], nodes[i + 1]

def build_edge_maps_from_df(link_df: pd.DataFrame) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    dirmap: Dict[Tuple[int, int], int] = {}
    undirmap: Dict[Tuple[int, int], int] = {}
    for _, row in link_df.iterrows():
        try:
            u = int(row['시작 노드'])
            v = int(row['끝 노드'])
            lid = int(row['LINK_ID'])
            dirmap[(u, v)] = lid
            undirmap[(u, v)] = lid
            undirmap[(v, u)] = lid  # 무방향 접근
        except Exception:
            continue
    return dirmap, undirmap

def nodes_to_link_ids(nodes: List[int], dirmap: Dict[Tuple[int, int], int], undirmap: Dict[Tuple[int, int], int]) -> List[int]:
    links: List[int] = []
    for u, v in pairwise_nodes(nodes):
        lid = dirmap.get((u, v))
        if lid is None:
            lid = undirmap.get((u, v))
        if lid is not None:
            links.append(lid)
    return links

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

# --- Lane penalty helpers ---
def lane_penalty_from_link_ids(link_ids: List[int]) -> Tuple[int, float]:
    """
    LaneChangeCalculator를 사용해 차선 이벤트 수와 벌점을 산출.
    벌점: 1/(1+n) (n=이벤트 수), 결과는 0~1 범위.
    """
    if not link_ids:
        return 0, 0.0
    try:
        res = lane_calc.calculate_lane_changes(link_ids)
        n = int(res.get("total_lane_changes", 0))
        pen = 1.0 / (1.0 + max(0, n))
        return n, float(pen)
    except Exception:
        return 0, 0.0

# === 성능 패치: KD-Tree 기반 최근접 노드 ===
def nearest_node_id_by_xy(point_xy: List[float]) -> Optional[int]:
    """
    point_xy: [x(lon), y(lat)]
    KD-Tree가 준비되지 않았으면 기존 find_closest_node_id로 폴백.
    """
    if node_kdtree is None or node_id_array is None:
        return find_closest_node_id(point_xy, diff_node)
    dist, idx = node_kdtree.query([point_xy[0], point_xy[1]], k=1)
    try:
        return int(node_id_array[idx])
    except Exception:
        return find_closest_node_id(point_xy, diff_node)

# === 성능 패치: k=1 최단경로는 간단 다익스트라 + LRU 캐시 ===
@lru_cache(maxsize=200_000)
def _shortest_nodes_tuple_cached(s_id: int, e_id: int) -> Tuple[int, ...]:
    try:
        nodes = nx.shortest_path(G, s_id, e_id, weight="eff_weight")
        return tuple(nodes)
    except Exception:
        return tuple()

def map_segment_to_link_ids(start_pt: List[float], end_pt: List[float]) -> List[int]:
    """
    세그먼트(시작/끝 좌표)를 그래프에 스냅 → (k=1) 최단경로 노드열 → 링크ID 열로 변환
    (KD-Tree + LRU 캐시로 가속)
    """
    s_id = nearest_node_id_by_xy(start_pt)
    e_id = nearest_node_id_by_xy(end_pt)
    if not (s_id and e_id):
        return []
    if s_id not in G or e_id not in G:
        return []
    nodes = list(_shortest_nodes_tuple_cached(int(s_id), int(e_id)))
    if not nodes:
        return []
    return nodes_to_link_ids(nodes, edge_dir_map, edge_undir_map)

def segment_lane_penalty(points: List[List[float]]) -> Tuple[int, float, List[int]]:
    if not points:
        return 0, 0.0, []
    start_pt = points[0]
    end_pt = points[-1]
    link_ids = map_segment_to_link_ids(start_pt, end_pt)
    cnt, pen = lane_penalty_from_link_ids(link_ids)
    return cnt, pen, link_ids

# --- 공공 속도 → 혼잡도 훅(없으면 0.5) ---
def public_congestion_from_links(link_ids: List[int]) -> float:
    """
    링크ID들의 공공 속도 값을 이용해 0~1 혼잡도로 환산(단순 훅).
    데이터 없으면 0.5 반환.
    """
    if not link_ids:
        return 0.5
    speeds = []
    for lid in link_ids:
        v = PUBLIC_SPEEDS.get(int(lid))
        if v is not None and np.isfinite(v) and v >= 0:
            speeds.append(float(v))
    if not speeds:
        return 0.5
    ref = np.percentile(speeds, 85) if len(speeds) >= 5 else max(speeds)
    if ref <= 0:
        return 0.5
    avg = sum(speeds) / len(speeds)
    cong = 1.0 - (avg / ref)
    if cong < 0.0: cong = 0.0
    if cong > 1.0: cong = 1.0
    return float(cong)

# ===================== FastAPI 모델 =====================
class Priority(str, Enum):
    distance = "DISTANCE"
    time = "TIME"
    recommend = "RECOMMEND"
    easy = "EASY"
    main_road = "MAIN_ROAD"
    no_traffic_info = "NO_TRAFFIC_INFO"

class Point(BaseModel):
    x: float  # 경도
    y: float  # 위도

class RequestBody(BaseModel):
    origin: Point
    destination: Point
    priority: Priority = Priority.recommend
    waypoints: Optional[List[Point]] = []

# ===================== lifespan =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global G, node_coords_map, difficulty_scorer, diff_node, diff_link, lane_calc
    global edge_dir_map, edge_undir_map
    global node_kdtree, node_id_array, node_xy_array

    logger.info("[BOOT] CSV 로드 시작")
    try:
        diff_node_file = r"C:\Users\user\python_project\senior_beginner_navigation_pr\busan_node_with_difficulty_all.csv"
        diff_link_file = r"C:\Users\user\python_project\senior_beginner_navigation_pr\busan_link_with_difficulty_all.csv"

        G, node_coords_map, diff_node, diff_link = create_graph_from_csv(
            node_file_path=diff_node_file,
            link_file_path=diff_link_file
        )
        logger.info(f"[BOOT] CSV 로드 완료: nodes={len(node_coords_map)}, edges={G.number_of_edges()}")

        ensure_edge_metrics(G, node_coords_map)
        normalize_edge_weights(G, node_coords_map, method="p95")
        edge_dir_map, edge_undir_map = build_edge_maps_from_df(diff_link)

        # 정적 스코어러 준비
        difficulty_scorer = DifficultyScorer(
            node_df=diff_node,
            link_df=diff_link,
            node_id_col="NODE_ID",
            node_x_col="경도",
            node_y_col="위도",
            node_diff_col="node_difficulty",
            link_u_col="시작 노드",
            link_v_col="끝 노드",
            link_diff_col="link_difficulty",
        )

        # 차선 이벤트 계산기
        lane_calc = LaneChangeCalculator(
            link_file=diff_link_file,
            node_file=diff_node_file
        )

        # === 성능 패치: KD-Tree 구성 (부팅 시 1회) ===
        node_xy_array = diff_node[["경도", "위도"]].to_numpy(dtype=float)  # [lon, lat]
        node_id_array = diff_node["NODE_ID"].to_numpy()
        node_kdtree = cKDTree(node_xy_array)
        logger.info(f"[BOOT] KD-Tree built: nodes={len(node_id_array)}")

    except Exception as e:
        logger.exception("데이터 로드/초기화 실패")
        raise RuntimeError(f"초기화 실패: {e}")

    logger.info("[BOOT] 서버 시작 준비 완료!")
    yield
    logger.info("[BOOT] 서버 종료 이벤트 발생: 자원 정리...")

# ===================== FastAPI 앱 =====================
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 외부 API =====================
def call_kakao(origin, destination, **kwargs):
    KAKAO_URL = "https://apis-navi.kakaomobility.com/v1/waypoints/directions"
    KAKAO_REST_KEY = os.getenv("KAKAO_REST_KEY", "c5ebe192c3d007a46ea0deb05f0ea12e").strip()
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}", "Content-Type": "application/json"}

    # ✅ priority를 kwargs에서 '꺼내서' 정규화 (여기서만 처리)
    raw_priority = kwargs.pop("priority", None)
    if isinstance(raw_priority, Enum):
        raw_priority = raw_priority.value
    if raw_priority is not None:
        raw_priority = str(raw_priority).upper()
        # 제휴 API에서 지원하는 우선순위 화이트리스트
        allowed = {"RECOMMEND", "TIME", "DISTANCE", "MAIN_ROAD", "NO_TRAFFIC_INFO"}
        if raw_priority not in allowed:
            logger.warning(f"[KAKAO] unsupported priority={raw_priority} -> RECOMMEND fallback")
            raw_priority = "RECOMMEND"

    payload = {"origin": origin, "destination": destination}
    if raw_priority:
        payload["priority"] = raw_priority

    # 나머지 옵션(waypoints, alternatives, road_details, summary 등)
    payload.update(kwargs)

    r = requests.post(KAKAO_URL, headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

#@app.get("/", response_class=HTMLResponse)
#async def read_html():
#    with open(r"C:\Users\user\python_project\senior_beginner_navigation_pr\DIVE-PROME\backend\test_index.html", "r", encoding="utf-8") as f:
#        html_content = f.read()
#    return html_content

#(추가) 간단한 헬스체크
@app.get("/health")
async def health():
    return {"ok": True}

# ===================== 엔드포인트 =====================
@app.post("/find-path")
async def find_path(body: RequestBody):
    origin_coords = {"x": body.origin.x, "y": body.origin.y}
    destination_coords = {"x": body.destination.x, "y": body.destination.y}
    waypoints_coords = [{"x": w.x, "y": w.y} for w in body.waypoints]

    t0 = time.time()
    logger.info(f"[REQ] priority={body.priority} origin=({body.origin.x:.6f},{body.origin.y:.6f}) "
                f"dest=({body.destination.x:.6f},{body.destination.y:.6f}) waypoints={len(waypoints_coords)}")

    # EASY가 아니면 카카오 원본 반환
    if body.priority != Priority.easy:
        kakao_json = call_kakao(
            origin_coords,
            destination_coords,
            priority=body.priority.value,   # Enum -> 문자열
            waypoints=waypoints_coords,
            alternatives=True,
            road_details=True
        )
        return kakao_json

    # EASY: 큰길 우선 경로들을 밑재료로 받아서 우리 '쉬운 길' 후처리
    kakao_json = call_kakao(
        origin_coords,
        destination_coords,
        priority="MAIN_ROAD",              # ← 제휴 옵션 그대로 사용
        waypoints=waypoints_coords,
        alternatives=True,
        road_details=True,
        summary=False
    )
    logger.info(f"[KAKAO] routes={len(kakao_json.get('routes', []))} road_details={bool(kakao_json.get('routes'))}")
    if not kakao_json or not kakao_json.get("routes"):
        raise HTTPException(status_code=404, detail="No route found from Kakao API.")

    # 1) 초기탐색: 베이스(정적+카카오속도) 스코어
    WEIGHTS_BASE = InitialWeights(
        w_congestion=0.5,
        w_static=0.5,
        w_node_in_static=0.6,
        w_link_in_static=0.4
    )
    ranked_base = difficulty_scorer.score_routes_initial_base(kakao_json, weights=WEIGHTS_BASE)

    # === 성능 패치: lane penalty/세그먼트 후처리는 상위 N개만 ===
    TOP_N_FOR_LANE = 1  # 종전: 모든 대안 경로 → 패치: 상위 1개만 정밀 평가
    candidates_to_enrich = ranked_base[:TOP_N_FOR_LANE]

    # 2) 카카오 각 경로: 세그먼트별 "정적+카카오속도+차선"으로 재합산(거리 가중)
    enriched: List[Dict[str, Any]] = []
    for i, rb in enumerate(candidates_to_enrich):
        per_road = rb["debug"]["per_road"]
        route_num = 0.0
        route_den = 0.0
        per_road_combined = []
        route_link_ids_all: List[int] = []

        for seg_idx, seg in enumerate(per_road):
            dist_m = float(seg.get("distance_m", 0.0))
            static_raw = float(seg.get("static_raw", 0.5))
            cong_raw = float(seg.get("congestion_raw", 0.5))  # speed 기반
            points = seg.get("points") or []

            # 세그먼트 차선 벌점 (KD-Tree+캐시 포함)
            lane_cnt, lane_pen, seg_link_ids = segment_lane_penalty(points)
            route_link_ids_all.extend(seg_link_ids)

            combined = WEIGHTS_BASE.w_static * static_raw + \
                       WEIGHTS_BASE.w_congestion * cong_raw + \
                       W_LANE * lane_pen
            combined = max(0.0, min(1.0, combined))

            route_num += combined * dist_m
            route_den += dist_m

            per_road_combined.append({
                "distance_m": dist_m,
                "difficulty_combined": combined,
                "static_raw": static_raw,
                "congestion_raw": cong_raw,
                "lane_changes": lane_cnt,
                "lane_penalty": round(lane_pen, 4),
                "points": points,
                "seg_link_ids": seg_link_ids,
            })

        route_combined = route_num / max(route_den, 1.0)
        enriched.append({
            "id": rb["id"],
            "distanceKm": rb["distanceKm"],
            "etaMin": rb["etaMin"],
            "difficulty": round(route_combined, 4),
            "debug": {
                **rb["debug"],
                "per_road_combined": per_road_combined,
                "route_link_ids": route_link_ids_all,
                "weights": {
                    "w_static": WEIGHTS_BASE.w_static,
                    "w_congestion": WEIGHTS_BASE.w_congestion,
                    "w_lane": W_LANE
                }
            },
            "path": rb["path"],
        })

    enriched.sort(key=lambda r: r["difficulty"])
    best_route = enriched[0]
    original_path_points = best_route["path"]
    final_path_points = list(original_path_points)  # 기본값
    final_source = "initial"

    kakao_total_len_m = float(best_route.get("debug", {}).get("dist_km", 0.0)) * 1000.0

    # 3) 위험구간 선정: "세그먼트 combined" 기준 임계 초과
    difficult_segments = []
    for seg in best_route["debug"]["per_road_combined"]:
        if seg["difficulty_combined"] > SEGMENT_DIFFICULTY_THRESHOLD:
            pts = seg["points"]
            difficult_segments.append({
                "start_point": pts[0],
                "end_point": pts[-1],
                "original_difficulty": float(seg["difficulty_combined"]),  # 이미 정적+카카오속도+차선
                "orig_lane_changes": seg["lane_changes"],
                "orig_lane_penalty": seg["lane_penalty"],
                "seg_link_ids": seg["seg_link_ids"],
                "distance_m": seg["distance_m"],
            })
    logger.info(f"[LOCAL] threshold={SEGMENT_DIFFICULTY_THRESHOLD} difficult_segments={len(difficult_segments)}")

    # 4) 세그먼트 우회 후보 생성(K=3) + 공공 속도/차선 반영 후 교체 판정
    local_path_points = list(original_path_points)
    local_rerouted = False
    reroute_logs: List[Dict[str, Any]] = []

    if difficult_segments:
        filtered_node_df = filter_nodes_by_radius(diff_node, difficult_segments, radius_km=5.0)
        logger.info(f"[LOCAL] 위험구간 주변 노드 필터 radius_km=5.0 -> 후보노드={len(filtered_node_df)}")

        if not filtered_node_df.empty:
            for segment in difficult_segments:
                start_point = segment['start_point']
                end_point   = segment['end_point']

                # ★ 여기서는 반경 필터를 유지하기 위해 기존 함수 사용(부분 집합 탐색)
                start_node_id = find_closest_node_id(start_point, filtered_node_df)
                end_node_id   = find_closest_node_id(end_point,   filtered_node_df)

                if not (start_node_id and end_node_id):
                    logger.info(f"[LOCAL][seg] 노드 매핑 실패 start_node={start_node_id} end_node={end_node_id}")
                    continue
                if start_node_id == end_node_id:
                    logger.info(f"[LOCAL][seg] 동일 노드(start=end={start_node_id}) → 스킵")
                    continue

                orig_score = float(segment["original_difficulty"])  # 이미 정적+카카오속도+차선
                orig_len_m = float(segment["distance_m"])

                cand_paths = k_alt_paths_between(G, start_node_id, end_node_id, k=K_SEGMENT)
                if not cand_paths:
                    logger.info("[LOCAL][seg] 후보 0개 → 스킵")
                    continue

                best_cand = None  # {"nodes","score","len","len_ratio","lane_changes","lane_pen","cong_raw"}

                for cidx, nodes in enumerate(cand_paths):
                    # 정적 평균
                    static_avg, total_len_m, _ = path_avg_difficulty(G, nodes, node_coords_map)
                    if not np.isfinite(static_avg) or total_len_m <= 0:
                        logger.info(f"[LOCAL][seg][cand#{cidx}] static/len 비정상 → 스킵")
                        continue

                    # 링크ID → 공공 혼잡
                    link_ids = nodes_to_link_ids(nodes, edge_dir_map, edge_undir_map)
                    cong_raw = public_congestion_from_links(link_ids)  # 0~1

                    # 차선 벌점
                    lane_cnt, lane_pen = lane_penalty_from_link_ids(link_ids)

                    # 후보 최종 점수(정적 + 공공속도 + 차선)
                    cand_score = WEIGHTS_BASE.w_static * static_avg + \
                                 WEIGHTS_BASE.w_congestion * cong_raw + \
                                 W_LANE * lane_pen
                    cand_score = max(0.0, min(1.0, cand_score))

                    len_ratio = (total_len_m / orig_len_m) if orig_len_m > 0 else float('inf')

                    logger.info(
                        f"[LOCAL][seg][cand#{cidx}] static_avg={static_avg:.4f} cong={cong_raw:.3f} "
                        f"lane_chg={lane_cnt} lane_pen={lane_pen:.3f} score={cand_score:.4f} "
                        f"len={total_len_m:.1f}m len_ratio={len_ratio:.3f} guard<={MAX_LENGTH_INCREASE_RATIO_LOCAL}"
                    )

                    # 길이 가드 + 개선
                    if (len_ratio <= MAX_LENGTH_INCREASE_RATIO_LOCAL) and (cand_score < orig_score):
                        if (best_cand is None) or (cand_score < best_cand["score"]) or \
                            (abs(cand_score - best_cand["score"]) < 1e-6 and total_len_m < best_cand["len"]):
                            best_cand = {
                                "nodes": nodes,
                                "score": cand_score,
                                "len": total_len_m,
                                "len_ratio": len_ratio,
                                "lane_changes": lane_cnt,
                                "lane_pen": lane_pen,
                                "cong_raw": cong_raw
                            }

                did_replace = False
                reason = "no_candidate_passed_guard_or_improved"
                chosen = None

                if best_cand:
                    try:
                        easy_path_points = build_path_points_from_nodes(best_cand["nodes"], node_coords_map)
                        if easy_path_points:
                            local_path_points = replace_segment(local_path_points, start_point, end_point, easy_path_points)
                            local_rerouted = True
                            did_replace = True
                            reason = "replaced: best_candidate"
                            chosen = best_cand
                        else:
                            reason = "easy_path_points_empty"
                    except ValueError:
                        reason = "replace_index_error(fuzzy-needed)"

                logger.info(
                    f"[LOCAL][seg] orig_score={orig_score:.4f} "
                    f"cand_score={(chosen['score'] if chosen else float('nan')):.4f} "
                    f"decision={reason} replaced={did_replace}"
                )

                reroute_logs.append({
                    "start_node": start_node_id,
                    "end_node": end_node_id,
                    "orig_score": round(orig_score, 4),
                    "orig_len_m": round(orig_len_m, 1),
                    "cand_score": round(float(chosen["score"]), 4) if chosen else None,
                    "cand_len_m": round(float(chosen["len"]), 1) if chosen else None,
                    "length_ratio": round(float(chosen["len_ratio"]), 3) if chosen else None,
                    "length_guard_limit": MAX_LENGTH_INCREASE_RATIO_LOCAL,
                    "cand_lane_changes": int(chosen["lane_changes"]) if chosen else None,
                    "cand_lane_penalty": round(float(chosen["lane_pen"]), 3) if chosen else None,
                    "cand_congestion_raw": round(float(chosen["cong_raw"]), 3) if chosen else None,
                    "replaced": did_replace,
                    "reason": reason
                })

    if local_rerouted:
        final_path_points = local_path_points
        final_source = "local"

    elapsed = (time.time() - t0) * 1000.0
    logger.info(f"[DONE] source={final_source} path_points={len(final_path_points)} elapsed={elapsed:.1f}ms")

    # === 여기부터 추가 (기존 EASY return 덩어리는 제거/교체) ===
    EASY_SAMPLING_PARAMS = SamplingParams(
        epsilon_base=35.0,   # 정식 필드명 ( *_m 아님 )
        alpha=1.0,
        beta=0.55,
        d_min=120.0,
        d_max=800.0
    )
    easy_sampling = build_easy_path_waypoints(final_path_points, params=EASY_SAMPLING_PARAMS)

    resp = {
        "path_points": final_path_points,        # [[lon,lat], ...]
        "source": final_source,                  # "initial" | "local"
        "local_rerouted": local_rerouted,
        "debug": {
            "ranked_base_top": ranked_base[:3],
            "best_initial": {
                "id": best_route["id"],
                "difficulty": best_route["difficulty"],
                "weights": best_route["debug"]["weights"],
            },
            "kakao_total_len_m": kakao_total_len_m,
            "difficult_segments": difficult_segments,
            "local_reroute_logs": reroute_logs,
            # "rerouted_segments": rerouted_segments,  # (옵션) 파란표시용
        },
        "request_echo": {
            "origin": {"x": body.origin.x, "y": body.origin.y},
            "destination": {"x": body.destination.x, "y": body.destination.y},
            "priority": body.priority.value if isinstance(body.priority, Enum) else str(body.priority)
        }
    }

    # EASY라면 샘플링 섹션 포함
    if body.priority == Priority.easy and final_path_points:
        resp["sampling"] = {
            "start": f"{final_path_points[0][1]},{final_path_points[0][0]}",
            "end": f"{final_path_points[-1][1]},{final_path_points[-1][0]}",
            "params": {
                "max_points": 30,
                "epsilon_base": 35.0,
                "alpha": 1.0,
                "beta": 0.55,
                "theta_turn": 25,
                "theta_uturn": 135,
                "d_min": 120,
                "d_max": 800
            },
            "routes": { "EASY_PATH": easy_sampling }
        }

    return resp
# ===================== main =====================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

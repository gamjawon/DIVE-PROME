# app.py
import os, json, requests
import pandas as pd
import numpy as np
import networkx as nx
import math, time, logging
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dataclasses import dataclass
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import cKDTree
from functools import lru_cache
import statistics

# --- 로거 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("navi")

# --- 외부 모듈 ---
from difficulty_scorer import DifficultyScorer, InitialWeights
from graph_builder import create_graph_from_csv
from utils import find_closest_node_id, filter_nodes_by_radius, replace_segment
from cal_lane_chng import LaneChangeCalculator

# ===================== 전역 =====================
G: nx.Graph = nx.Graph()
node_coords_map: Dict[int, Tuple[float, float]] = {}
difficulty_scorer: DifficultyScorer
diff_node: pd.DataFrame
diff_link: pd.DataFrame
lane_calc: LaneChangeCalculator

edge_dir_map: Dict[Tuple[int, int], int] = {}
edge_undir_map: Dict[Tuple[int, int], int] = {}

PUBLIC_SPEEDS: Dict[int, float] = {}

MAX_LENGTH_INCREASE_RATIO_LOCAL = 50.0
K_SEGMENT = 3
SEGMENT_DIFFICULTY_THRESHOLD = 0.95
W_LANE = 0.20

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
    epsilon_base_m: Optional[float] = None
    d_min_m: Optional[float] = None
    d_max_m: Optional[float] = None
    def __post_init__(self):
        if self.epsilon_base_m is not None: self.epsilon_base = float(self.epsilon_base_m)
        if self.d_min_m is not None: self.d_min = float(self.d_min_m)
        if self.d_max_m is not None: self.d_max = float(self.d_max_m)
        if self.epsilon_base_m is None: self.epsilon_base_m = float(self.epsilon_base)
        if self.d_min_m is None: self.d_min_m = float(self.d_min)
        if self.d_max_m is None: self.d_max_m = float(self.d_max)

@dataclass
class SamplingWeights:
    w_turn: float = 0.35
    w_uturn: float = 0.25
    w_lane: float = 0.20
    w_intersection: float = 0.10
    w_class: float = 0.05
    w_hazard: float = 0.05

def _kakao_first_route_polyline_lonlat(kakao_json) -> List[List[float]]:
    pts = []
    try:
        routes = kakao_json.get("routes", []) or []
        if not routes: return []
        rt = routes[0]
        for sec in rt.get("sections", []) or []:
            for rd in sec.get("roads", []) or []:
                vx = rd.get("vertexes", []) or []
                for i in range(0, len(vx) - 1, 2):
                    pts.append([float(vx[i]), float(vx[i+1])])  # [lon,lat]
    except Exception:
        pass
    return pts

def waypoint_from_link_id_midpoint(link_id: int) -> Optional[Dict[str, float]]:
    try:
        row = diff_link.loc[diff_link["LINK_ID"] == int(link_id)].iloc[0]
        lon1, lat1 = float(row["링크시작경도"]), float(row["링크시작위도"])
        lon2, lat2 = float(row["링크끝경도"]), float(row["링크끝위도"])
        mx = (lon1 + lon2) / 2.0
        my = (lat1 + lat2) / 2.0
        return {"x": mx, "y": my}
    except Exception:
        logger.warning(f"[WAYPOINT] LINK_ID {link_id} midpoint not found")
        return None

def _count_uturns_from_path(points_lonlat: List[List[float]], theta_uturn: float = 135.0) -> int:
    import math
    n = len(points_lonlat)
    if n < 3: return 0
    def angle_deg(a, b, c) -> float:
        ax, ay = float(a[0]), float(a[1]); bx, by = float(b[0]), float(b[1]); cx, cy = float(c[0]), float(c[1])
        v1x, v1y = bx - ax, by - ay; v2x, v2y = cx - bx, cy - by
        n1 = (v1x*v1x + v1y*v1y) ** 0.5; n2 = (v2x*v2x + v2y*v2y) ** 0.5
        if n1 == 0.0 or n2 == 0.0: return 0.0
        cosv = (v1x*v2x + v1y*v2y) / (n1*n2); cosv = max(-1.0, min(1.0, cosv))
        return math.degrees(math.acos(cosv))
    return sum(1 for i in range(1, n-1) if angle_deg(points_lonlat[i-1], points_lonlat[i], points_lonlat[i+1]) >= float(theta_uturn))

def _kakao_summary(kakao_json) -> Tuple[int, int]:
    try:
        routes = kakao_json.get("routes", []) or []
        if not routes: return 0, 0
        sm = routes[0].get("summary", {}) or {}
        return int(float(sm.get("duration", 0))), int(float(sm.get("distance", 0)))
    except Exception:
        return 0, 0

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def edge_length_m(u: int, v: int, node_xy: Dict[int, Tuple[float, float]]) -> float:
    lon1, lat1 = node_xy[u]; lon2, lat2 = node_xy[v]
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
        if w is not None and np.isfinite(w): vals.append(float(w))
    if not vals: return
    p95 = np.percentile(vals, 95)
    if not np.isfinite(p95) or p95 <= 0: p95 = max(vals) or 1.0
    for u, v, data in graph.edges(data=True):
        raw = float(data.get("weight", 0.0))
        data["weight_raw"] = raw
        norm = raw / p95
        norm = 0.0 if norm < 0.0 else (1.0 if norm > 1.0 else norm)
        data["weight"] = norm
        if "length_m" not in data:
            data["length_m"] = edge_length_m(u, v, node_xy)
        data["eff_weight"] = max(data["length_m"], 1.0) * data["weight"]

def path_avg_difficulty(graph: nx.Graph, path_nodes: List[int], node_xy: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float]:
    if not path_nodes or len(path_nodes) < 2: return float('inf'), 0.0, float('inf')
    total_len = 0.0; weighted_sum = 0.0
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]; v = path_nodes[i + 1]
        if u not in node_xy or v not in node_xy: return float('inf'), 0.0, float('inf')
        data = graph.get_edge_data(u, v) or {}
        length_m = data.get('length_m') or edge_length_m(u, v, node_xy)
        link_diff = float(data.get('weight', 0.0))
        total_len += length_m; weighted_sum += link_diff * length_m
    if total_len <= 0: return float('inf'), 0.0, float('inf')
    return weighted_sum / total_len, total_len, weighted_sum

def build_path_points_from_nodes(path_nodes: List[int], node_xy: Dict[int, Tuple[float, float]]) -> List[List[float]]:
    pts = []
    for n in path_nodes:
        if n in node_xy:
            x, y = node_xy[n]; pts.append([float(x), float(y)])  # [lon,lat]
    return pts

def pairwise_nodes(nodes: List[int]):
    for i in range(len(nodes) - 1):
        yield nodes[i], nodes[i + 1]

def build_edge_maps_from_df(link_df: pd.DataFrame) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    dirmap: Dict[Tuple[int, int], int] = {}
    undirmap: Dict[Tuple[int, int], int] = {}
    for _, row in link_df.iterrows():
        try:
            u = int(row['시작 노드']); v = int(row['끝 노드']); lid = int(row['LINK_ID'])
            dirmap[(u, v)] = lid; undirmap[(u, v)] = lid; undirmap[(v, u)] = lid
        except Exception:
            continue
    return dirmap, undirmap

def nodes_to_link_ids(nodes: List[int], dirmap: Dict[Tuple[int, int], int], undirmap: Dict[Tuple[int, int], int]) -> List[int]:
    links: List[int] = []
    for u, v in pairwise_nodes(nodes):
        lid = dirmap.get((u, v))
        if lid is None: lid = undirmap.get((u, v))
        if lid is not None: links.append(lid)
    return links

def k_alt_paths_between(graph: nx.Graph, source: int, target: int, k: int = 5) -> List[List[int]]:
    try:
        gen = nx.shortest_simple_paths(graph, source, target, weight="eff_weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    paths = []
    for i, p in enumerate(gen):
        if i >= k: break
        paths.append(p)
    return paths

# lane helpers
@lru_cache(maxsize=200_000)
def _shortest_nodes_tuple_cached(s_id: int, e_id: int) -> Tuple[int, ...]:
    try:
        nodes = nx.shortest_path(G, s_id, e_id, weight="eff_weight")
        return tuple(nodes)
    except Exception:
        return tuple()

def nearest_node_id_by_xy(point_xy: List[float]) -> Optional[int]:
    if node_kdtree is None or node_id_array is None:
        return find_closest_node_id(point_xy, diff_node)
    dist, idx = node_kdtree.query([point_xy[0], point_xy[1]], k=1)
    try:
        return int(node_id_array[idx])
    except Exception:
        return find_closest_node_id(point_xy, diff_node)

def map_segment_to_link_ids(start_pt: List[float], end_pt: List[float]) -> List[int]:
    s_id = nearest_node_id_by_xy(start_pt); e_id = nearest_node_id_by_xy(end_pt)
    if not (s_id and e_id): return []
    if s_id not in G or e_id not in G: return []
    nodes = list(_shortest_nodes_tuple_cached(int(s_id), int(e_id)))
    if not nodes: return []
    return nodes_to_link_ids(nodes, edge_dir_map, edge_undir_map)

def lane_penalty_from_link_ids(link_ids: List[int]) -> Tuple[int, float]:
    if not link_ids: return 0, 0.0
    try:
        res = lane_calc.calculate_lane_changes(link_ids)
        n = int(res.get("total_lane_changes", 0))
        pen = 1.0 / (1.0 + max(0, n))
        return n, float(pen)
    except Exception:
        return 0, 0.0

def lane_changes_for_polyline(points: List[List[float]]) -> int:
    # points: [[lon,lat], ...]
    link_ids: List[int] = []
    for i in range(len(points)-1):
        link_ids.extend(map_segment_to_link_ids(points[i], points[i+1]))
    n, _ = lane_penalty_from_link_ids(link_ids)
    return int(n)

# ---- 샘플링/평활(시각화용 거의 직선) ----
def _angle_deg(a, b, c):
    import math
    bax = a[0] - b[0]; bay = a[1] - b[1]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]
    na = math.hypot(bax, bay); nb = math.hypot(bcx, bcy)
    if na == 0 or nb == 0: return 0.0
    cosv = max(-1.0, min(1.0, (bax*bcx + bay*bcy) / (na*nb)))
    return math.degrees(math.acos(cosv))

def _segment_len_m(p, q): return haversine_m(p[1], p[0], q[1], q[0])

def _perp_dist_m(p, a, b):
    ax, ay = a[0], a[1]; bx, by = b[0], b[1]; px, py = p[0], p[1]
    if ax == bx and ay == by: return _segment_len_m(p, a)
    vx, vy = bx - ax, by - ay; wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    t = max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
    proj = [ax + t*vx, ay + t*vy]
    return _segment_len_m(p, proj)

def _extract_features(points, theta_turn=25.0, theta_uturn=135.0):
    n = len(points); turns = [0.0]*n; is_uturn = [0]*n; is_intersection = [0]*n; lane_change=[0]*n; class_change=[0]*n; hazard=[0.0]*n
    seg_len = [0.0]*(n-1)
    for i in range(n-1): seg_len[i] = _segment_len_m(points[i], points[i+1])
    for i in range(1, n-1):
        ang = _angle_deg(points[i-1], points[i], points[i+1]); turns[i] = ang
        if ang >= theta_uturn: is_uturn[i] = 1
        if ang >= theta_turn: is_intersection[i] = 1
        if 15 <= ang <= 60:
            left = seg_len[i-1] if i-1 < len(seg_len) else 0
            right = seg_len[i] if i < len(seg_len) else 0
            if left < 80 or right < 80: lane_change[i] = 1
    return {"turns":turns,"is_uturn":is_uturn,"is_intersection":is_intersection,"lane_change":lane_change,"class_change":class_change,"hazard":hazard,"seg_len":seg_len}

def _score_importance(feat, weights: SamplingWeights, theta_turn_clip=90.0):
    turns = feat["turns"]; n = len(turns); imp = [0.0]*n
    for i in range(n):
        f_turn = min(1.0, abs(turns[i]) / theta_turn_clip)
        imp[i] = (weights.w_turn*f_turn + weights.w_uturn*feat["is_uturn"][i] + weights.w_lane*feat["lane_change"][i] +
                  weights.w_intersection*feat["is_intersection"][i] + weights.w_class*feat["class_change"][i] +
                  weights.w_hazard*feat["hazard"][i])
    return imp

def _required_indices(n, feat, imp, tau=0.6):
    req = set()
    if n>0: req.add(0)
    if n>1: req.add(n-1)
    for i in range(n):
        if feat["is_uturn"][i] or feat["is_intersection"][i] or feat["lane_change"][i] or imp[i] >= tau:
            req.add(i)
    return req

def _weighted_rdp(points, imp, params: SamplingParams, keep:set):
    n = len(points)
    if n <= 2: return list(range(n)), []
    kept = set(keep); removed=set()
    def rdp(l, r):
        a = points[l]; b = points[r]
        max_d=-1.0; idx=-1
        for i in range(l+1, r):
            eps_i = params.epsilon_base / (1.0 + params.alpha*imp[i])
            d = _perp_dist_m(points[i], a, b) - eps_i
            if d > max_d: max_d = d; idx = i
        if max_d > 0:
            rdp(l, idx); rdp(idx, r)
        else:
            for i in range(l+1, r):
                if i not in keep: removed.add(i)
            kept.add(l); kept.add(r)
    rdp(0, n-1)
    kept_idx = sorted(list(kept)); removed_idx = sorted(list(removed - kept))
    return kept_idx, removed_idx

def _adaptive_pacing(points, kept_idx, imp, params: SamplingParams):
    n = len(points)
    if n == 0: return []
    kept_idx = sorted(set(kept_idx))
    if kept_idx[0] != 0: kept_idx = [0] + kept_idx
    if kept_idx[-1] != n-1: kept_idx = kept_idx + [n-1]
    cur = len(kept_idx)
    if cur >= params.max_points: return kept_idx[:params.max_points]
    segs=[]
    for a,b in zip(kept_idx[:-1], kept_idx[1:]):
        d=0.0
        for i in range(a,b): d += _segment_len_m(points[i], points[i+1])
        g=sum(imp[a:b+1]); segs.append({"a":a,"b":b,"D":d,"G":g})
    sumD=sum(s["D"] for s in segs) or 1.0; sumG=sum(s["G"] for s in segs) or 1.0
    for s in segs:
        pD=s["D"]/sumD; pG=s["G"]/sumG; s["P"]=0.55*pG+0.45*pD
    L = params.max_points - cur
    alloc=[max(0, round(L*s["P"])) for s in segs]
    new_idx=set(kept_idx)
    for s,cnt in zip(segs, alloc):
        if cnt<=0: continue
        a,b=s["a"], s["b"]
        seg_len_m=s["D"]; 
        if seg_len_m<=0: continue
        targets=[(k+1)/(cnt+1)*seg_len_m for k in range(cnt)]
        cum=0.0; t_idx=0
        for i in range(a,b):
            edge=_segment_len_m(points[i], points[i+1])
            while t_idx < len(targets) and cum+edge >= targets[t_idx]:
                cand=i+1
                if cand not in new_idx: new_idx.add(cand)
                t_idx+=1
            cum+=edge
    out=sorted(list(new_idx))
    # d_min 보장
    pruned=[]; last=None; dmin=params.d_min_m or params.d_min
    for idx in out:
        if last is None: pruned.append(idx); last=idx
        else:
            if _segment_len_m(points[last], points[idx]) >= dmin:
                pruned.append(idx); last=idx
    return pruned[:params.max_points]
def segment_lane_penalty(points: List[List[float]]) -> Tuple[int, float, List[int]]:
    """
    Kakao 한 세그먼트(polyline)의 시작/끝을 그래프에 스냅해서
    링크ID 시퀀스를 추정하고, 차선변경 수/벌점을 계산한다.
    반환: (lane_changes_count, lane_penalty, link_ids)
    """
    if not points:
        return 0, 0.0, []
    start_pt = points[0]           # [lon,lat]
    end_pt   = points[-1]          # [lon,lat]
    link_ids = map_segment_to_link_ids(start_pt, end_pt)
    n, pen   = lane_penalty_from_link_ids(link_ids)
    return n, pen, link_ids
def build_easy_path_waypoints(path_points: List[List[float]],
                              params: SamplingParams = SamplingParams(),
                              weights: SamplingWeights = SamplingWeights()):
    n = len(path_points)
    feat = _extract_features(path_points, params.theta_turn, params.theta_uturn)
    imp = _score_importance(feat, weights)
    req = _required_indices(n, feat, imp, tau=0.6)
    kept_rdp, _ = _weighted_rdp(path_points, imp, params, keep=req)
    idx = _adaptive_pacing(path_points, kept_rdp, imp, params)
    path_30 = [[float(path_points[i][1]), float(path_points[i][0])] for i in idx]  # [lat,lng]
    total_dist=0.0
    for i in range(n-1): total_dist += _segment_len_m(path_points[i], path_points[i+1])
    lane_changes = sum(feat["lane_change"]); u_turns = sum(feat["is_uturn"])
    vals = [imp[i] for i in idx]
    score_stats = {"min": round(min(vals), 4) if vals else 0.0,
                   "p50": round(statistics.median(vals), 4) if vals else 0.0,
                   "p90": round(np.percentile(vals, 90), 4) if vals else 0.0,
                   "max": round(max(vals), 4) if vals else 0.0}
    keyword = "안전경로"
    if lane_changes <= 1: keyword += "|차선변경적음"
    if u_turns == 0: keyword += "|유턴없음"
    return {
        "path_point_raw_count": n,
        "path_point_30": path_30,
        "keyword": keyword,
        "duration": None,
        "distance": round(total_dist, 1),
        "lane_changes": int(lane_changes),
        "u_turns": int(u_turns),
        "debug": {"score_stats": score_stats}
    }

def simplify_for_display_lonlat(points_lonlat: List[List[float]]) -> List[List[float]]:
    """시각화용 강력 평활: 거의 직선 느낌으로 간결화된 polyline 반환([[lon,lat], ...])"""
    if not points_lonlat: return []
    STRONG = SamplingParams(
        max_points=20,        # 더 적은 꼭짓점
        epsilon_base=120.0,   # 수선 거리 허용 크게
        alpha=1.0,
        beta=0.4,
        d_min=180.0,          # 최소 간격 ↑
        d_max=1500.0,
        theta_turn=35.0,
        theta_uturn=150.0
    )
    smp = build_easy_path_waypoints(points_lonlat, params=STRONG)
    latlng = smp.get("path_point_30", []) or []
    # [lat,lng] -> [lon,lat]
    return [[lng, lat] for (lat, lng) in latlng]

# ===================== FastAPI 모델 =====================
class Point(BaseModel):
    x: float  # 경도
    y: float  # 위도

class RequestBody(BaseModel):
    origin: Point
    destination: Point
    # (priority/날씨 입력 제거)
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

        difficulty_scorer = DifficultyScorer(
            node_df=diff_node, link_df=diff_link,
            node_id_col="NODE_ID", node_x_col="경도", node_y_col="위도",
            node_diff_col="node_difficulty",
            link_u_col="시작 노드", link_v_col="끝 노드", link_diff_col="link_difficulty",
        )

        lane_calc = LaneChangeCalculator(link_file=diff_link_file, node_file=diff_node_file)

        node_xy_array = diff_node[["경도", "위도"]].to_numpy(dtype=float)
        node_id_array = diff_node["NODE_ID"].to_numpy()
        node_kdtree = cKDTree(node_xy_array)
        logger.info(f"[BOOT] KD-Tree built: nodes={len(node_id_array)}")

    except Exception as e:
        logger.exception("데이터 로드/초기화 실패")
        raise RuntimeError(f"초기화 실패: {e}")

    logger.info("[BOOT] 서버 시작 준비 완료!")
    yield
    logger.info("[BOOT] 서버 종료 이벤트 발생: 자원 정리...")

# ===================== 앱/외부 API =====================
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def call_kakao(origin, destination, **kwargs):
    KAKAO_URL = "https://apis-navi.kakaomobility.com/v1/waypoints/directions"
    KAKAO_REST_KEY = os.getenv("KAKAO_REST_KEY", "c5ebe192c3d007a46ea0deb05f0ea12e").strip()
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}", "Content-Type": "application/json"}

    raw_priority = kwargs.pop("priority", None)
    if raw_priority is not None:
        raw_priority = str(raw_priority).upper()
        allowed = {"RECOMMEND", "TIME", "DISTANCE", "MAIN_ROAD", "NO_TRAFFIC_INFO"}
        if raw_priority not in allowed:
            logger.warning(f"[KAKAO] unsupported priority={raw_priority} -> RECOMMEND")
            raw_priority = "RECOMMEND"

    payload = {"origin": origin, "destination": destination}
    if raw_priority: payload["priority"] = raw_priority
    payload.update(kwargs)

    r = requests.post(KAKAO_URL, headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

@app.get("/", response_class=HTMLResponse)
async def read_html():
    with open(r"C:\Users\user\python_project\senior_beginner_navigation_pr\DIVE-PROME\backend\test_index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health(): return {"ok": True}

# ---------- 공통 빌더 ----------
WEIGHTS_BASE = InitialWeights(
    w_congestion=0.5, w_static=0.5, w_node_in_static=0.6, w_link_in_static=0.4
)

def build_route_result_from_kakao(kakao_json: Dict[str, Any], label: str) -> Dict[str, Any]:
    """카카오 결과를 path_points/요약/차선변경/유턴/표시용 경로까지 패키징."""
    path_points = _kakao_first_route_polyline_lonlat(kakao_json)  # [[lon,lat], ...]
    duration_sec, distance_m = _kakao_summary(kakao_json)
    # lane changes: per_road 기준으로 링크 매핑
    ranked = difficulty_scorer.score_routes_initial_base(kakao_json, weights=WEIGHTS_BASE)
    rb0 = ranked[0] if ranked else None
    route_link_ids: List[int] = []
    if rb0:
        for seg in rb0["debug"]["per_road"]:
            pts = seg.get("points") or []
            if len(pts) >= 2:
                _, _, lnk = segment_lane_penalty(pts)
                route_link_ids.extend(lnk)
    lane_changes, _ = lane_penalty_from_link_ids(route_link_ids)
    uturns = _count_uturns_from_path(path_points)

    display_points = simplify_for_display_lonlat(path_points)
    return {
        "label": label,
        "duration_sec": int(duration_sec),
        "distance_m": int(distance_m),
        "u_turns": int(uturns),
        "lane_changes": int(lane_changes),
        "path_points": path_points,                 # 원본 [[lon,lat], ...]
        "display_path_points": display_points,      # 시각화용 평활 경로
        "kakao_final": kakao_json                   # 필요시 프런트에서 그대로 그림
    }

# ===================== 엔드포인트: 3종 한방 계산 =====================
@app.post("/find-path")
async def find_path(body: RequestBody):
    t0 = time.time()
    origin_coords = {"x": body.origin.x, "y": body.origin.y}
    destination_coords = {"x": body.destination.x, "y": body.destination.y}
    waypoints_coords = [{"x": w.x, "y": w.y} for w in (body.waypoints or [])]

    logger.info(f"[REQ] origin=({body.origin.x:.6f},{body.origin.y:.6f}) "
                f"dest=({body.destination.x:.6f},{body.destination.y:.6f})")

    # 1) RECOMMEND
    rec_json = call_kakao(origin_coords, destination_coords,
                          priority="RECOMMEND", waypoints=waypoints_coords,
                          alternatives=True, road_details=True)
    rec_bundle = build_route_result_from_kakao(rec_json, "RECOMMEND")

    # 2) MAIN_ROAD
    main_json = call_kakao(origin_coords, destination_coords,
                           priority="MAIN_ROAD", waypoints=waypoints_coords,
                           alternatives=True, road_details=True)
    main_bundle = build_route_result_from_kakao(main_json, "MAIN_ROAD")

    # 3) EASY: MAIN_ROAD 밑재료로 후처리
    #    (기존 EASY 로직을 경량화: 위험 구간만 대체하고 재호출 없음)
    routes = main_json.get("routes") or []
    if not routes: raise HTTPException(status_code=404, detail="No route from Kakao (MAIN_ROAD).")
    rt = routes[0]; secs = rt.get("sections", []) or []
    logger.info(f"[EASY] sections={len(secs)}")

    ranked_base = difficulty_scorer.score_routes_initial_base(main_json, weights=WEIGHTS_BASE)
    # lane/seg 보강
    TOP_N_FOR_LANE = 1
    candidates_to_enrich = ranked_base[:TOP_N_FOR_LANE]
    enriched: List[Dict[str, Any]] = []
    for rb in candidates_to_enrich:
        per_road = rb["debug"]["per_road"]
        route_num = 0.0; route_den = 0.0; per_road_combined = []; route_link_ids_all=[]
        for seg in per_road:
            dist_m = float(seg.get("distance_m", 0.0))
            static_raw = float(seg.get("static_raw", 0.5))
            cong_raw = float(seg.get("congestion_raw", 0.5))
            points = seg.get("points") or []
            lane_cnt, lane_pen, seg_link_ids = segment_lane_penalty(points)
            route_link_ids_all.extend(seg_link_ids)
            combined = WEIGHTS_BASE.w_static*static_raw + WEIGHTS_BASE.w_congestion*cong_raw + W_LANE*lane_pen
            combined = max(0.0, min(1.0, combined))
            route_num += combined * dist_m; route_den += dist_m
            per_road_combined.append({
                "distance_m": dist_m, "difficulty_combined": combined,
                "static_raw": static_raw, "congestion_raw": cong_raw,
                "lane_changes": lane_cnt, "lane_penalty": round(lane_pen,4),
                "points": points, "seg_link_ids": seg_link_ids
            })
        route_combined = route_num / max(route_den, 1.0)
        enriched.append({
            "id": rb["id"], "distanceKm": rb["distanceKm"], "etaMin": rb["etaMin"],
            "difficulty": round(route_combined, 4),
            "debug": {**rb["debug"], "per_road_combined": per_road_combined,
                      "route_link_ids": route_link_ids_all,
                      "weights": {"w_static":WEIGHTS_BASE.w_static, "w_congestion":WEIGHTS_BASE.w_congestion, "w_lane":W_LANE}},
            "path": rb["path"],
        })
    enriched.sort(key=lambda r: r["difficulty"])
    best_route = enriched[0]
    original_path_points = best_route["path"]
    local_path_points = list(original_path_points)

    difficult_segments = []
    for seg in best_route["debug"]["per_road_combined"]:
        if seg["difficulty_combined"] > SEGMENT_DIFFICULTY_THRESHOLD:
            pts = seg["points"]
            difficult_segments.append({
                "start_point": pts[0], "end_point": pts[-1],
                "original_difficulty": float(seg["difficulty_combined"]),
                "seg_link_ids": seg["seg_link_ids"], "distance_m": seg["distance_m"],
            })
    if difficult_segments:
        filtered_node_df = filter_nodes_by_radius(diff_node, difficult_segments, radius_km=5.0)
        if not filtered_node_df.empty:
            for segment in difficult_segments:
                start_point = segment['start_point']; end_point = segment['end_point']
                start_node_id = find_closest_node_id(start_point, filtered_node_df)
                end_node_id = find_closest_node_id(end_point, filtered_node_df)
                if not (start_node_id and end_node_id): continue
                if start_node_id == end_node_id: continue
                orig_score = float(segment["original_difficulty"]); orig_len_m = float(segment["distance_m"])
                cand_paths = k_alt_paths_between(G, start_node_id, end_node_id, k=K_SEGMENT)
                best_cand=None
                for nodes in cand_paths:
                    static_avg, total_len_m, _ = path_avg_difficulty(G, nodes, node_coords_map)
                    if not np.isfinite(static_avg) or total_len_m<=0: continue
                    link_ids = nodes_to_link_ids(nodes, edge_dir_map, edge_undir_map)
                    _, lane_pen = lane_penalty_from_link_ids(link_ids)
                    cong_raw = 0.5
                    cand_score = WEIGHTS_BASE.w_static*static_avg + WEIGHTS_BASE.w_congestion*cong_raw + W_LANE*lane_pen
                    cand_score = max(0.0, min(1.0, cand_score))
                    len_ratio = (total_len_m / orig_len_m) if orig_len_m > 0 else float('inf')
                    if (len_ratio <= MAX_LENGTH_INCREASE_RATIO_LOCAL) and (cand_score < orig_score):
                        if (best_cand is None) or (cand_score < best_cand["score"]) or \
                           (abs(cand_score - best_cand["score"]) < 1e-6 and total_len_m < best_cand["len"]):
                            best_cand={"nodes":nodes,"score":cand_score,"len":total_len_m}
                if best_cand:
                    easy_points = build_path_points_from_nodes(best_cand["nodes"], node_coords_map)
                    if easy_points:
                        local_path_points = replace_segment(local_path_points, start_point, end_point, easy_points)

    # EASY 결과 패키징
    easy_path_points = local_path_points
    easy_duration_sec = int(best_route.get("etaMin", 0)*60)
    easy_distance_m = int(float(best_route.get("debug", {}).get("dist_km", 0.0))*1000.0)

    easy_lane_changes = lane_changes_for_polyline(easy_path_points)
    easy_uturns = _count_uturns_from_path(easy_path_points)
    easy_display = simplify_for_display_lonlat(easy_path_points)  # ✅ 강력 평활(거의 직선)

    easy_bundle = {
        "label": "EASY",
        "duration_sec": int(easy_duration_sec),
        "distance_m": int(easy_distance_m),
        "u_turns": int(easy_uturns),
        "lane_changes": int(easy_lane_changes),
        "path_points": easy_path_points,            # 원본
        "display_path_points": easy_display,        # 시각화용(강력 평활)
        "kakao_final": None
    }

    elapsed_ms = (time.time() - t0) * 1000.0
    logger.info(f"[DONE] all routes built in {elapsed_ms:.1f}ms")

    return {
        "routes": {
            "RECOMMEND": rec_bundle,
            "MAIN_ROAD": main_bundle,
            "EASY": easy_bundle
        },
        "request_echo": {
            "origin": {"x": body.origin.x, "y": body.origin.y},
            "destination": {"x": body.destination.x, "y": body.destination.y}
        },
        "elapsed_ms": round(elapsed_ms, 1)
    }

# ===================== main =====================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

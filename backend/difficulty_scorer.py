from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


@dataclass
class InitialWeights:
    # 동일 척도 조합(카카오/공공 모두 동일)
    w_static: float = 0.5
    w_congestion: float = 0.4
    w_lane: float = 0.1
    # 정적 내부
    w_node_in_static: float = 0.6
    w_link_in_static: float = 0.4
    # 차선 정규화(벌점 계산용)
    lane_norm: int = 3


class DifficultyScorer:
    """
    초기 경로의 '베이스 난이도'만 계산 (정적 + 속도 기반 정체)
    - 차선 변경 벌점은 LaneChangeCalculator를 이용해 app.py에서 합산한다.
    - traffic_state는 사용하지 않고 traffic_speed만 사용한다.
    - 성능 패치:
        * 최근접 노드 탐색: cKDTree (O(log N))
        * 링크 난이도 조회: (u,v) → link_difficulty 해시맵
    """

    def __init__(
        self,
        node_df: pd.DataFrame,
        link_df: pd.DataFrame,
        node_id_col: str = "NODE_ID",
        node_x_col: str = "경도",
        node_y_col: str = "위도",
        node_diff_col: str = "node_difficulty",
        link_u_col: str = "시작 노드",
        link_v_col: str = "끝 노드",
        link_diff_col: str = "link_difficulty",
    ):
        # 원본/메타
        self.node_df = node_df
        self.link_df = link_df

        # 컬럼명
        self.node_id_col = node_id_col
        self.node_x_col = node_x_col
        self.node_y_col = node_y_col
        self.node_diff_col = node_diff_col
        self.link_u_col = link_u_col
        self.link_v_col = link_v_col
        self.link_diff_col = link_diff_col

        # --- 노드 배열 & KD-Tree ---
        self._node_xy = self.node_df[[self.node_x_col, self.node_y_col]].to_numpy(dtype=float)  # [lon, lat]
        self._node_ids = self.node_df[self.node_id_col].to_numpy()
        self._node_diff = self.node_df[self.node_diff_col].to_numpy(dtype=float)
        self._kdtree = cKDTree(self._node_xy) if len(self._node_xy) > 0 else None

        # --- 링크 난이도 해시맵: (u,v) & (v,u) → difficulty ---
        # df 조회(필터) 비용 제거
        self._link_diff_map: Dict[Tuple[int, int], float] = {}
        if not self.link_df.empty:
            cols = [self.link_u_col, self.link_v_col, self.link_diff_col]
            for _, row in self.link_df[cols].dropna(subset=[self.link_u_col, self.link_v_col, self.link_diff_col]).iterrows():
                try:
                    u = int(row[self.link_u_col])
                    v = int(row[self.link_v_col])
                    d = float(row[self.link_diff_col])
                    self._link_diff_map[(u, v)] = d
                    self._link_diff_map[(v, u)] = d  # 무방향 접근 허용
                except Exception:
                    continue

    # --------------- Kakao 유틸 ---------------
    @staticmethod
    def _vertexes_to_points(vertexes_1d: List[float]) -> List[List[float]]:
        pts: List[List[float]] = []
        for i in range(0, len(vertexes_1d), 2):
            if i + 1 < len(vertexes_1d):
                pts.append([float(vertexes_1d[i]), float(vertexes_1d[i + 1])])  # [lon, lat]
        return pts

    # --------------- 최근접 노드(KD-Tree) ---------------
    def _nearest_node_idx(self, x: float, y: float) -> int:
        """노드 배열이 비어있지 않다는 전제. KD-Tree가 없으면 안전 폴백."""
        if self._kdtree is not None:
            _, idx = self._kdtree.query([x, y], k=1)
            return int(idx)
        # 폴백: 선형탐색(아주 드문 경우만)
        dx = (self._node_xy[:, 0] - x)
        dy = (self._node_xy[:, 1] - y)
        dist2 = dx * dx + dy * dy
        return int(np.argmin(dist2))

    # --------------- 정체(속도) ---------------
    @staticmethod
    def _congestion_from_speed(traffic_speed: Optional[float], ref_speed: Optional[float]) -> Tuple[float, Dict[str, Any]]:
        """
        혼잡도(0~1) = clamp(1 - speed/ref, 0, 1)
        """
        if traffic_speed is None or ref_speed is None or ref_speed <= 0:
            return 0.5, {"mode": "none"}
        try:
            sp = float(traffic_speed)
            sev = max(0.0, min(1.0, 1.0 - sp / float(ref_speed)))
            return sev, {"mode": "speed", "speed": sp, "ref": float(ref_speed)}
        except Exception:
            return 0.5, {"mode": "none"}

    # --------------- 정적 스코어 ---------------
    def _link_diff_between(self, u_id: int, v_id: int) -> float:
        """사전에서 즉시 조회, 없으면 0.5."""
        val = self._link_diff_map.get((u_id, v_id))
        if val is None:
            val = self._link_diff_map.get((v_id, u_id))
        return float(val) if val is not None else 0.5

    def _static_score_for_road(self, road: Dict[str, Any], w_node: float, w_link: float) -> Tuple[float, Dict[str, Any]]:
        """
        근사: 도로 중심에 가장 가까운 노드의 node_difficulty,
            시작/끝점에 가까운 노드 2개로 링크를 조회해 link_difficulty를 얻고
            w_node, w_link로 가중평균
        (성능 패치: KD-Tree 사용 + 링크난이도 해시조회)
        """
        pts = road.get("points") or []
        if not pts:
            return 0.5, {"mode": "empty"}

        # 중심점
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        try:
            idx_center = self._nearest_node_idx(cx, cy)
            node_d = float(self._node_diff[idx_center])
            near_node = int(self._node_ids[idx_center])
        except Exception:
            node_d = 0.5
            near_node = None

        # 링크 난이도 (시작/끝의 최근접 노드 → (u,v))
        link_d = 0.5
        try:
            sx, sy = pts[0]
            ex, ey = pts[-1]
            i_s = self._nearest_node_idx(sx, sy)
            i_e = self._nearest_node_idx(ex, ey)
            u = int(self._node_ids[i_s])
            v = int(self._node_ids[i_e])
            link_d = self._link_diff_between(u, v)
        except Exception:
            pass

        static = max(0.0, min(1.0, w_node * node_d + w_link * link_d))
        return static, {"node_id": near_node, "node_diff": node_d, "link_diff": link_d}

    # --------------- 초기(베이스) 스코어링 ---------------
    def score_routes_initial_base(
        self,
        kakao_json: Dict[str, Any],
        weights: InitialWeights = InitialWeights(),
    ) -> List[Dict[str, Any]]:
        """
        경로별 '베이스 난이도' (정적 + 카카오 traffic_speed 기반 정체)
        ※ 차선 벌점은 여기서 계산하지 않음 (LaneChangeCalculator로 app.py에서 합산)
        """
        routes = kakao_json.get("routes", []) or []
        results: List[Dict[str, Any]] = []

        for i, rt in enumerate(routes):
            # ref_speed = 85th percentile
            speeds: List[float] = []
            for sec in rt.get("sections", []) or []:
                for rd in sec.get("roads", []) or []:
                    v = rd.get("traffic_speed")
                    if v is not None:
                        try:
                            vv = float(v)
                            if vv > 0:
                                speeds.append(vv)
                        except Exception:
                            pass
            ref_speed: Optional[float] = float(np.percentile(speeds, 85)) if speeds else None

            per_road: List[Dict[str, Any]] = []
            num = 0.0
            den = 0.0
            path_pts: List[List[float]] = []

            for sec in rt.get("sections", []) or []:
                for rd in sec.get("roads", []) or []:
                    vx = rd.get("vertexes", []) or []
                    points = self._vertexes_to_points(vx)
                    # 전체 경로 좌표 저장
                    for p in points:
                        path_pts.append(p)

                    distance_m = float(rd.get("distance", 0.0))
                    road_obj = {
                        "points": points,
                        "traffic_speed": rd.get("traffic_speed"),
                        "distance_m": distance_m,
                    }

                    cong_raw, cong_dbg = self._congestion_from_speed(road_obj["traffic_speed"], ref_speed)
                    static_raw, static_dbg = self._static_score_for_road(
                        road=road_obj,
                        w_node=weights.w_node_in_static,
                        w_link=weights.w_link_in_static,
                    )

                    diff_base = max(
                        0.0,
                        min(
                            1.0,
                            weights.w_congestion * float(cong_raw)
                            + weights.w_static * float(static_raw),
                        ),
                    )

                    num += diff_base * distance_m
                    den += distance_m

                    per_road.append({
                        "distance_m": distance_m,
                        "difficulty": diff_base,             # (정적+정체)만
                        "congestion_raw": cong_raw, "congestion_dbg": cong_dbg,
                        "static_raw": static_raw, "static_dbg": static_dbg,
                        "points": points,
                    })

            route_base = num / max(den, 1.0)
            sm = rt.get("summary", {}) or {}
            results.append({
                "id": (sm.get("priority", "route") or "route") + f"_{i+1}",
                "distanceKm": float(sm.get("distance", 0.0)) / 1000.0,
                "etaMin": float(sm.get("duration", 0.0)) / 60.0,
                "base_difficulty": round(float(route_base), 4),   # ← 차선 제외
                "debug": {
                    "ref_speed": ref_speed,
                    "dist_km": round(den / 1000.0, 3),
                    "per_road": per_road,
                    "weights": {
                        "w_static": weights.w_static,
                        "w_congestion": weights.w_congestion,
                        "w_node_in_static": weights.w_node_in_static,
                        "w_link_in_static": weights.w_link_in_static,
                    }
                },
                "path": path_pts,
            })

        results.sort(key=lambda r: r["base_difficulty"])
        return results

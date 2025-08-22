import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2, degrees

class LaneChangeCalculator:
    def __init__(self, link_file, node_file):
        """
        차선 변경 횟수 계산기 초기화
        
        Args:
            link_file: 링크 데이터 CSV 파일 경로
            node_file: 노드 데이터 CSV 파일 경로
        """
        # 데이터 로드 시 결측값 처리
        self.link_df = pd.read_csv(link_file).dropna(subset=['LINK_ID'])
        self.node_df = pd.read_csv(node_file).dropna(subset=['NODE_ID'])
        
        # 숫자형 데이터 타입 확인 및 변환
        self.link_df['LINK_ID'] = pd.to_numeric(self.link_df['LINK_ID'], errors='coerce')
        self.node_df['NODE_ID'] = pd.to_numeric(self.node_df['NODE_ID'], errors='coerce')
        self.link_df['시작 노드'] = pd.to_numeric(self.link_df['시작 노드'], errors='coerce')
        self.link_df['끝 노드'] = pd.to_numeric(self.link_df['끝 노드'], errors='coerce')
        
        # 결측값 제거
        self.link_df = self.link_df.dropna(subset=['LINK_ID', '시작 노드', '끝 노드'])
        self.node_df = self.node_df.dropna(subset=['NODE_ID'])
        
        # 빠른 조회를 위한 딕셔너리 생성
        self.link_dict = self.link_df.set_index('LINK_ID').to_dict('index')
        self.node_dict = self.node_df.set_index('NODE_ID').to_dict('index')
        
        print(f"데이터 로드 완료: 링크 {len(self.link_df)}개, 노드 {len(self.node_df)}개")
        
        # 데이터 정합성 체크
        self._check_data_integrity()
    
    def _check_data_integrity(self):
        """데이터 정합성 체크"""
        # 링크의 노드 ID들이 노드 데이터에 존재하는지 확인
        all_node_ids = set(self.node_df['NODE_ID'].astype(int))
        start_nodes = set(self.link_df['시작 노드'].astype(int))
        end_nodes = set(self.link_df['끝 노드'].astype(int))
        
        missing_start = start_nodes - all_node_ids
        missing_end = end_nodes - all_node_ids
        
        if missing_start or missing_end:
            print(f"경고: 일부 노드 정보가 누락되었습니다.")
            print(f"누락된 시작 노드: {len(missing_start)}개")
            print(f"누락된 끝 노드: {len(missing_end)}개")
    
    def calculate_bearing(self, lon1, lat1, lon2, lat2):
        """두 점 사이의 방향각 계산 (라디안)"""
        try:
            lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            
            y = sin(dlon) * cos(lat2)
            x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
            
            return atan2(y, x)
        except (ValueError, TypeError):
            return 0
    
    def angle_difference(self, angle1, angle2):
        """각도 차이 계산 (절댓값, 도 단위)"""
        try:
            diff = abs(degrees(angle1) - degrees(angle2))
            if diff > 180:
                diff = 360 - diff
            return diff
        except (ValueError, TypeError):
            return 0
    
    def calculate_distance(self, lon1, lat1, lon2, lat2):
        """두 좌표 사이의 거리 계산 (미터)"""
        try:
            lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371000  # 지구 반지름 (미터)
            
            return c * r
        except (ValueError, TypeError):
            return float('inf')
    
    def detect_turn_events(self, route):
        """회전 이벤트 감지 (30도 이상 회전)"""
        turn_events = []
        
        for i in range(1, len(route) - 1):
            try:
                prev_link = self.link_dict.get(int(route[i-1]))
                curr_link = self.link_dict.get(int(route[i]))
                next_link = self.link_dict.get(int(route[i+1]))
                
                if not all([prev_link, curr_link, next_link]):
                    continue
                
                # 좌표 정보 확인
                coord_keys = ['링크시작경도', '링크시작위도', '링크끝경도', '링크끝위도']
                if not all(key in prev_link and key in curr_link and key in next_link for key in coord_keys):
                    continue
                
                # 각 링크의 방향각 계산
                bearing1 = self.calculate_bearing(
                    prev_link['링크시작경도'], prev_link['링크시작위도'],
                    prev_link['링크끝경도'], prev_link['링크끝위도']
                )
                bearing2 = self.calculate_bearing(
                    curr_link['링크시작경도'], curr_link['링크시작위도'],
                    curr_link['링크끝경도'], curr_link['링크끝위도']
                )
                bearing3 = self.calculate_bearing(
                    next_link['링크시작경도'], next_link['링크시작위도'],
                    next_link['링크끝경도'], next_link['링크끝위도']
                )
                
                # 연속된 방향각의 차이 계산
                angle1 = self.angle_difference(bearing1, bearing2)
                angle2 = self.angle_difference(bearing2, bearing3)
                max_angle = max(angle1, angle2)
                
                if max_angle >= 30:
                    turn_events.append({
                        'link_index': i,
                        'link_id': int(route[i]),
                        'angle': max_angle,
                        'type': 'turn',
                        'reason': f'{max_angle:.1f}도 회전'
                    })
            except (KeyError, TypeError, ValueError) as e:
                # 디버깅: 어떤 에러가 발생하는지 출력
                # print(f"Turn event 감지 중 에러 (링크 {i}): {e}")
                continue
        
        return turn_events
    
    def detect_grade_changes(self, route):
        """도로 등급 변화 감지 (2등급 이상 차이)"""
        grade_events = []
        
        for i in range(1, len(route)):
            try:
                prev_link = self.link_dict.get(int(route[i-1]))
                curr_link = self.link_dict.get(int(route[i]))
                
                if not all([prev_link, curr_link]):
                    continue
                
                prev_grade = prev_link.get('도로등급', 0)
                curr_grade = curr_link.get('도로등급', 0)
                
                if prev_grade is None or curr_grade is None:
                    continue
                
                grade_diff = abs(float(prev_grade) - float(curr_grade))
                
                if grade_diff >= 2:
                    grade_events.append({
                        'link_index': i,
                        'link_id': int(route[i]),
                        'grade_diff': grade_diff,
                        'type': 'grade_change',
                        'reason': f"도로등급 변화 ({prev_grade} → {curr_grade})"
                    })
            except (KeyError, TypeError, ValueError) as e:
                # print(f"Grade change 감지 중 에러 (링크 {i}): {e}")
                continue
        
        return grade_events
    
    def detect_lane_changes(self, route):
        """차로수 변화 감지 (2차선 이상 차이)"""
        lane_events = []
        
        for i in range(1, len(route)):
            try:
                prev_link = self.link_dict.get(int(route[i-1]))
                curr_link = self.link_dict.get(int(route[i]))
                
                if not all([prev_link, curr_link]):
                    continue
                
                prev_lanes = prev_link.get('차로수', 0)
                curr_lanes = curr_link.get('차로수', 0)
                
                if prev_lanes is None or curr_lanes is None:
                    continue
                
                lane_diff = abs(float(prev_lanes) - float(curr_lanes))
                
                if lane_diff >= 2:
                    lane_events.append({
                        'link_index': i,
                        'link_id': int(route[i]),
                        'lane_diff': lane_diff,
                        'type': 'lane_change',
                        'reason': f"차로수 급변 ({prev_lanes}차선 → {curr_lanes}차선)"
                    })
            except (KeyError, TypeError, ValueError) as e:
                # print(f"Lane change 감지 중 에러 (링크 {i}): {e}")
                continue
        
        return lane_events
    
    def detect_intersection_complexity(self, route):
        """교차로 복잡도 감지 (연결도 4 이상)"""
        intersection_events = []
        processed_nodes = set()
        
        for i, link_id in enumerate(route):
            try:
                link = self.link_dict.get(int(link_id))
                if not link:
                    continue
                
                start_node_id = link.get('시작 노드')
                end_node_id = link.get('끝 노드')
                
                for node_id in [start_node_id, end_node_id]:
                    if node_id is None or pd.isna(node_id) or node_id in processed_nodes:
                        continue
                    
                    try:
                        node_id = int(node_id)
                        node = self.node_dict.get(node_id)
                        
                        if not node:
                            continue
                        
                        node_degree = node.get('node_degree', 0)
                        if node_degree is None:
                            continue
                        
                        if float(node_degree) >= 4:
                            intersection_events.append({
                                'link_index': i,
                                'link_id': int(link_id),
                                'node_id': node_id,
                                'degree': float(node_degree),
                                'type': 'intersection',
                                'reason': f"복잡 교차로 (연결도 {node_degree})"
                            })
                            processed_nodes.add(node_id)
                    except (KeyError, TypeError, ValueError):
                        continue
                        
            except (KeyError, TypeError, ValueError) as e:
                # print(f"Intersection 감지 중 에러 (링크 {i}): {e}")
                continue
        
        return intersection_events
    
    def merge_nearby_events(self, events, route):
        """근접한 이벤트 병합 (80m 이내)"""
        if not events:
            return events
        
        # 이벤트에 위치 정보 추가
        for event in events:
            try:
                link = self.link_dict.get(int(event['link_id']))
                if (link and 
                    '링크시작경도' in link and '링크시작위도' in link and
                    link['링크시작경도'] is not None and link['링크시작위도'] is not None):
                    event['position'] = {
                        'lon': float(link['링크시작경도']),
                        'lat': float(link['링크시작위도'])
                    }
            except (KeyError, TypeError, ValueError):
                continue
        
        merged_events = []
        processed = set()
        
        for i, event in enumerate(events):
            if i in processed:
                continue
            
            nearby_events = [event]
            processed.add(i)
            
            # 80m 이내의 다른 이벤트들 찾기
            for j in range(i + 1, len(events)):
                if j in processed:
                    continue
                
                other_event = events[j]
                if 'position' not in event or 'position' not in other_event:
                    continue
                
                try:
                    distance = self.calculate_distance(
                        event['position']['lon'], event['position']['lat'],
                        other_event['position']['lon'], other_event['position']['lat']
                    )
                    
                    if distance <= 80:
                        nearby_events.append(other_event)
                        processed.add(j)
                except (KeyError, TypeError, ValueError):
                    continue
            
            # 병합된 이벤트 생성
            if len(nearby_events) > 1:
                types = [e['type'] for e in nearby_events]
                merged_events.append({
                    **event,
                    'type': 'merged',
                    'reason': f"병합된 이벤트 ({len(nearby_events)}개): {', '.join(types)}",
                    'sub_events': nearby_events,
                    'merged_count': len(nearby_events)
                })
            else:
                merged_events.append(event)
        
        return merged_events
    
    def calculate_lane_changes(self, route):
        """경로의 차선 변경 횟수 계산"""
        # 입력 검증
        if not route or len(route) < 2:
            return {
                'total_lane_changes': 0,
                'events': [],
                'breakdown': {
                    'turns': 0,
                    'grade_changes': 0,
                    'lane_changes': 0,
                    'intersections': 0,
                    'merged': 0,
                    'original_total': 0
                }
            }
        
        # 각종 이벤트 감지
        turn_events = self.detect_turn_events(route)
        grade_events = self.detect_grade_changes(route)
        lane_events = self.detect_lane_changes(route)
        intersection_events = self.detect_intersection_complexity(route)
        
        # 모든 이벤트 합치기
        all_events = turn_events + grade_events + lane_events + intersection_events
        
        # 링크 인덱스 기준 정렬
        all_events.sort(key=lambda x: x['link_index'])
        
        # 근접한 이벤트 병합
        merged_events = self.merge_nearby_events(all_events, route)
        
        return {
            'total_lane_changes': len(merged_events),
            'events': merged_events,
            'breakdown': {
                'turns': len(turn_events),
                'grade_changes': len(grade_events),
                'lane_changes': len(lane_events),
                'intersections': len(intersection_events),
                'merged': len([e for e in merged_events if e['type'] == 'merged']),
                'original_total': len(all_events)
            }
        }
    
    def analyze_multiple_routes(self, routes):
        """여러 경로 분석"""
        results = []
        
        for i, route in enumerate(routes):
            print(f"\n=== 경로 {i + 1} 분석 ===")
            result = self.calculate_lane_changes(route)
            
            print(f"경로 길이: {len(route)}개 링크")
            print(f"총 차선 변경 횟수: {result['total_lane_changes']}회")
            print(f"병합 전 이벤트 수: {result['breakdown']['original_total']}회")
            
            if result['breakdown']['merged'] > 0:
                print(f"근접 이벤트 병합: {result['breakdown']['merged']}개 그룹")
            
            print("세부 분석:")
            print(f"- 회전 이벤트: {result['breakdown']['turns']}회")
            print(f"- 도로등급 변화: {result['breakdown']['grade_changes']}회")
            print(f"- 차로수 변화: {result['breakdown']['lane_changes']}회")
            print(f"- 교차로 복잡도: {result['breakdown']['intersections']}회")
            
            if result['events']:
                print("주요 이벤트 (상위 3개):")
                for j, event in enumerate(result['events'][:3]):
                    print(f"  {j + 1}. {event['reason']}")
            
            results.append({
                'route_index': i + 1,
                'route': route,
                **result
            })
        
        return results


# 사용 예시
def main():
    # 계산기 초기화
    calculator = LaneChangeCalculator(
        '/Users/gamjawon/busan_link_with_difficulty_all.csv',
        '/Users/gamjawon/busan_node_with_difficulty_all.csv'
    )
    
    # 예시 경로들 (실제로는 경로 알고리즘에서 받아올 데이터)
    # 여기서는 데이터의 첫 번째 몇 개 링크 ID를 사용
    sample_routes = [
        list(calculator.link_df['LINK_ID'].head(8)),      # 경로 1: 8개 링크
        list(calculator.link_df['LINK_ID'].iloc[500:508]), # 경로 2: 8개 링크
        list(calculator.link_df['LINK_ID'].iloc[1000:1010]) # 경로 3: 10개 링크
    ]
    
    print("=== 차선 변경 횟수 계산 결과 ===")
    
    # 다중 경로 분석
    results = calculator.analyze_multiple_routes(sample_routes)
    
    # 요약 결과
    print(f"\n=== 요약 ===")
    for result in results:
        print(f"경로 {result['route_index']}: {result['total_lane_changes']}회")
    
    return results

# 실행
if __name__ == "__main__":
    results = main()
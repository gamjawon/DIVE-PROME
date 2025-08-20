import pandas as pd
import requests
import json
import time
import numpy as np
from geopy.distance import geodesic
import re
import os
from datetime import datetime, timedelta

class RoadWidthIntegrator:
    def __init__(self, kakao_api_key):
        self.api_key = kakao_api_key
        self.headers = {"Authorization": f"KakaoAK {kakao_api_key}"}
        self.base_url = "https://dapi.kakao.com/v2/local"
        
        # API 호출 제한 관리
        self.daily_limit = 300000  # 일일 호출 제한
        self.per_second_limit = 10  # 초당 호출 제한
        self.call_count = 0
        self.start_time = datetime.now()
        self.last_call_time = 0
        
        # 진행상황 저장을 위한 변수
        self.checkpoint_interval = 100  # 100개마다 저장
        self.checkpoint_file = "checkpoint_progress.json"
        
    def load_checkpoint(self):
        """체크포인트 파일에서 진행상황 로드"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def save_checkpoint(self, index, results):
        """체크포인트 저장"""
        checkpoint_data = {
            'last_processed_index': index,
            'processed_count': len(results),
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def check_api_limits(self):
        """API 호출 제한 확인 및 대기"""
        current_time = time.time()
        
        # 초당 제한 확인
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < (1.0 / self.per_second_limit):
            sleep_time = (1.0 / self.per_second_limit) - time_since_last_call
            print(f"⏳ API 제한으로 {sleep_time:.2f}초 대기...")
            time.sleep(sleep_time)
        
        # 일일 제한 확인
        if self.call_count >= self.daily_limit:
            elapsed_time = datetime.now() - self.start_time
            if elapsed_time.total_seconds() < 86400:  # 24시간
                wait_time = 86400 - elapsed_time.total_seconds()
                print(f"🚫 일일 API 호출 제한 도달. {wait_time/3600:.1f}시간 후 재시작하세요.")
                return False
        
        self.last_call_time = time.time()
        self.call_count += 1
        return True
    
    def safe_api_call(self, url, params, max_retries=3):
        """안전한 API 호출 (재시도 로직 포함)"""
        for attempt in range(max_retries):
            try:
                # API 제한 확인
                if not self.check_api_limits():
                    return None
                
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Too Many Requests
                    wait_time = 2 ** attempt  # 지수 백오프
                    print(f"⚠️ API 제한 오류 (429). {wait_time}초 후 재시도... (시도 {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                elif response.status_code == 500:  # 서버 오류
                    wait_time = 5
                    print(f"⚠️ 서버 오류 (500). {wait_time}초 후 재시도... (시도 {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API 오류: {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⏰ 타임아웃 오류. 재시도... (시도 {attempt+1}/{max_retries})")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"🌐 네트워크 오류: {e}. 재시도... (시도 {attempt+1}/{max_retries})")
                time.sleep(2)
        
        return None
    
    def search_address_coordinates(self, road_name, sido="부산광역시", sigungu=""):
        """개선된 좌표 검색 (안전한 API 호출 사용)"""
        try:
            full_address = self.get_full_road_address(road_name, sido, sigungu)
            
            # 주소 검색 API 호출
            url = f"{self.base_url}/search/address.json"
            params = {
                'query': full_address,
                'analyze_type': 'similar'
            }
            
            data = self.safe_api_call(url, params)
            
            if data and data['documents']:
                result = data['documents'][0]
                return {
                    'longitude': float(result['x']),
                    'latitude': float(result['y']),
                    'address_name': result['address_name'],
                    'road_address_name': result.get('road_address_name', ''),
                    'search_query': full_address
                }
            
            # 주소 검색 실패시 키워드 검색 시도
            return self.search_keyword_coordinates(full_address)
            
        except Exception as e:
            print(f"주소 검색 실패 ({road_name}): {e}")
            return None
    
    def search_keyword_coordinates(self, keyword):
        """개선된 키워드 검색"""
        try:
            url = f"{self.base_url}/search/keyword.json"
            params = {
                'query': keyword,
                'category_group_code': 'FD6',
                'x': 129.0756,
                'y': 35.1796,
                'radius': 5000
            }
            
            data = self.safe_api_call(url, params)
            
            if data and data['documents']:
                result = data['documents'][0]
                return {
                    'longitude': float(result['x']),
                    'latitude': float(result['y']),
                    'address_name': result['address_name'],
                    'road_address_name': result.get('road_address_name', ''),
                    'place_name': result['place_name']
                }
            
        except Exception as e:
            print(f"키워드 검색 실패 ({keyword}): {e}")
            
        return None
    
    def create_coordinate_mapping_with_resume(self, road_width_df):
        """중단된 지점부터 재시작 가능한 좌표 매핑"""
        results = []
        start_index = 0
        
        # 체크포인트에서 복구
        checkpoint = self.load_checkpoint()
        if checkpoint:
            start_index = checkpoint['last_processed_index'] + 1
            results = checkpoint['results']
            print(f"🔄 체크포인트에서 복구: {start_index}번째부터 재시작")
            print(f"📊 기존 처리 완료: {len(results)}개")
        
        total_count = len(road_width_df)
        
        for idx in range(start_index, total_count):
            row = road_width_df.iloc[idx]
            
            # 진행상황 표시
            progress = ((idx + 1) / total_count) * 100
            estimated_remaining = (total_count - idx - 1) * 0.15  # 초당 약 6.7개 처리 가정
            
            print(f"Processing {idx+1}/{total_count}: {row['road_name']} ({row['sigungu']}) - {progress:.1f}% 완료, 예상 남은 시간: {estimated_remaining/60:.1f}분")
            
            # 좌표 검색
            coord_info = self.search_address_coordinates(
                road_name=row['road_name'],
                sido=row['sido'],
                sigungu=row['sigungu']
            )
            
            if coord_info:
                result_row = row.to_dict()
                result_row.update(coord_info)
                result_row['search_success'] = True
            else:
                result_row = row.to_dict()
                result_row.update({
                    'longitude': None,
                    'latitude': None,
                    'address_name': '',
                    'road_address_name': '',
                    'search_success': False
                })
            
            results.append(result_row)
            
            # 체크포인트 저장
            if (idx + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(idx, results)
                print(f"💾 체크포인트 저장: {idx + 1}개 처리 완료")
                
                # 중간 결과도 저장
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"temp_coordinate_mapping_{idx+1}.csv", index=False, encoding='utf-8-sig')
            
            # API 제한 확인
            if self.call_count >= self.daily_limit:
                print(f"🚫 일일 API 호출 제한 도달. 진행상황이 저장되었습니다.")
                print(f"📍 현재까지 {len(results)}개 처리 완료")
                break
        
        # 최종 체크포인트 저장
        if results:
            self.save_checkpoint(len(results) - 1, results)
        
        return pd.DataFrame(results)
    
    def get_api_usage_report(self):
        """API 사용량 보고서"""
        elapsed_time = datetime.now() - self.start_time
        calls_per_minute = (self.call_count / elapsed_time.total_seconds()) * 60 if elapsed_time.total_seconds() > 0 else 0
        
        print(f"\n📊 API 사용량 보고서:")
        print(f"  총 호출 수: {self.call_count}")
        print(f"  경과 시간: {elapsed_time}")
        print(f"  분당 호출 수: {calls_per_minute:.1f}")
        print(f"  남은 일일 호출량: {self.daily_limit - self.call_count}")

    def load_road_width_csv(self, csv_path):
        """
        부산시 도로구간조서 CSV 파일을 로드하고 파싱
        
        Args:
            csv_path (str): CSV 파일 경로
            
        Returns:
            pd.DataFrame: 파싱된 도로 폭 데이터
        """
        try:
            # 다양한 인코딩으로 시도
            encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig']
            df = None
            
            for encoding in encodings:
                try:
                    print(f"인코딩 {encoding}으로 파일 읽기 시도...")
                    df = pd.read_csv(csv_path, encoding=encoding)
                    print(f"✅ {encoding} 인코딩으로 파일 읽기 성공!")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"❌ {encoding} 인코딩 실패: {e}")
                    continue
            
            if df is None:
                raise Exception("모든 인코딩으로 파일 읽기 실패")
            
            print(f"📊 원본 데이터 정보:")
            print(f"  - 총 행 수: {len(df)}")
            print(f"  - 컬럼 수: {len(df.columns)}")
            print(f"  - 컬럼명: {list(df.columns)}")
            
            # 컬럼명 정리 (한글 깨짐 방지)
            df.columns = df.columns.str.strip()
            
            # 예상 컬럼명들 (인코딩 문제로 깨질 수 있음)
            possible_columns = {
                'sido': ['시도명', 'sido', '½Ãµµ¸í'],
                'sigungu': ['시군구명', 'sigungu', '½Ã±º±¸¸í'], 
                'road_class': ['도로위계', 'road_class', 'µµ·ÎÀ§°è'],
                'road_name': ['도로명', 'road_name', 'µµ·Î¸í'],
                'road_type': ['종속구분', 'road_type', 'Á¾¼Ó±¸ºÐ'],
                'length': ['연장', 'length', '¿¬Àå'],
                'width': ['폭', 'width', 'Æø']
            }
            
            # 컬럼 매핑
            column_mapping = {}
            for standard_name, possible_names in possible_columns.items():
                for col in df.columns:
                    if col in possible_names:
                        column_mapping[col] = standard_name
                        break
            
            print(f"📝 컬럼 매핑: {column_mapping}")
            
            # 컬럼명 변경
            df_renamed = df.rename(columns=column_mapping)
            
            # 필수 컬럼 확인
            required_cols = ['road_name', 'length', 'width']
            missing_cols = [col for col in required_cols if col not in df_renamed.columns]
            
            if missing_cols:
                print(f"⚠️ 필수 컬럼 누락: {missing_cols}")
                print("컬럼 순서 기반으로 매핑 시도...")
                
                # 컬럼 순서 기반 매핑 (시도명, 시군구명, 도로위계, 도로명, 종속구분, 연장, 폭)
                if len(df.columns) >= 7:
                    df_renamed = df.copy()
                    df_renamed.columns = ['sido', 'sigungu', 'road_class', 'road_name', 'road_type', 'length', 'width']
                    print("✅ 순서 기반 컬럼 매핑 완료")
            
            # 데이터 정제
            road_data = []
            
            for idx, row in df_renamed.iterrows():
                try:
                    # 필수 데이터 추출
                    road_name = str(row.get('road_name', '')).strip()
                    length = float(row.get('length', 0))
                    width = float(row.get('width', 0))
                    
                    # 유효한 데이터만 선택
                    if road_name and length > 0 and width > 0:
                        # 지역 정보 추가
                        sido = str(row.get('sido', '부산광역시')).strip()
                        sigungu = str(row.get('sigungu', '')).strip()
                        
                        road_data.append({
                            'road_id': str(idx + 1),  # 행 번호를 ID로 사용
                            'sido': sido,
                            'sigungu': sigungu,
                            'road_name': road_name,
                            'length': length,
                            'width': width,
                            'road_class': str(row.get('road_class', '')).strip(),
                            'road_type': str(row.get('road_type', '')).strip()
                        })
                        
                except (ValueError, TypeError) as e:
                    continue
            
            result_df = pd.DataFrame(road_data)
            
            print(f"🎉 데이터 파싱 완료:")
            print(f"  - 유효한 도로 수: {len(result_df)}")
            print(f"  - 평균 도로 폭: {result_df['width'].mean():.1f}m")
            print(f"  - 도로 폭 범위: {result_df['width'].min()}m ~ {result_df['width'].max()}m")
            
            # 샘플 데이터 출력
            print(f"\n📋 샘플 데이터:")
            for i, row in result_df.head().iterrows():
                print(f"  {i+1}. {row['road_name']} ({row['sigungu']}) - 연장: {row['length']}m, 폭: {row['width']}m")
            
            return result_df
            
        except Exception as e:
            print(f"❌ CSV 파일 로드 실패: {e}")
            raise
    
    def get_full_road_address(self, road_name, sido="부산광역시", sigungu=""):
        """
        도로명으로부터 완전한 주소 생성
        
        Args:
            road_name (str): 도로명
            sido (str): 시도명
            sigungu (str): 시군구명
            
        Returns:
            str: 완전한 주소
        """
        # 주소 구성
        if sigungu and sigungu != "":
            full_address = f"{sido} {sigungu} {road_name}"
        else:
            full_address = f"{sido} {road_name}"
        
        return full_address.strip()
    
    def calculate_distance(self, coord1, coord2):
        """
        두 좌표 간의 거리 계산 (미터)
        
        Args:
            coord1 (tuple): (latitude, longitude)
            coord2 (tuple): (latitude, longitude)
            
        Returns:
            float: 거리 (미터)
        """
        try:
            return geodesic(coord1, coord2).meters
        except:
            return float('inf')
    
    def create_spatial_grid(self, coords_df, grid_size=0.001):
        """
        공간 그리드 인덱스 생성 (성능 최적화용)
        
        Args:
            coords_df (pd.DataFrame): 좌표 데이터
            grid_size (float): 그리드 크기 (도 단위, 약 100m)
            
        Returns:
            dict: 그리드별 인덱스 딕셔너리
        """
        grid_index = {}
        
        for idx, row in coords_df.iterrows():
            if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                # 그리드 셀 계산
                grid_x = int(row['longitude'] / grid_size)
                grid_y = int(row['latitude'] / grid_size)
                grid_key = (grid_x, grid_y)
                
                if grid_key not in grid_index:
                    grid_index[grid_key] = []
                grid_index[grid_key].append(idx)
        
        return grid_index
    
    def get_nearby_grid_cells(self, lat, lon, grid_size=0.001, radius_cells=2):
        """
        주변 그리드 셀들 반환
        
        Args:
            lat, lon (float): 기준 좌표
            grid_size (float): 그리드 크기
            radius_cells (int): 검색 반경 (셀 개수)
            
        Returns:
            list: 주변 그리드 셀 키 리스트
        """
        center_x = int(lon / grid_size)
        center_y = int(lat / grid_size)
        
        nearby_cells = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nearby_cells.append((center_x + dx, center_y + dy))
        
        return nearby_cells
    
    def match_road_width_to_links(self, road_coord_df, link_df, max_distance=500):
        """
        공간 인덱싱을 사용한 최적화된 도로 폭 매칭
        
        Args:
            road_coord_df (pd.DataFrame): 좌표가 포함된 도로 폭 데이터
            link_df (pd.DataFrame): 링크 데이터
            max_distance (float): 최대 매칭 거리 (미터)
            
        Returns:
            pd.DataFrame: 도로 폭 정보가 추가된 링크 데이터
        """
        # 좌표 정보가 있는 도로 데이터만 사용
        valid_roads = road_coord_df[road_coord_df['search_success'] == True].copy()
        
        print(f"🚀 최적화된 매칭 시작: {len(valid_roads)}개 도로, {len(link_df)}개 링크")
        
        # 링크 데이터에 중점 좌표 추가
        link_df = link_df.copy()
        link_df['mid_latitude'] = (link_df['링크시작위도'] + link_df['링크끝위도']) / 2
        link_df['mid_longitude'] = (link_df['링크시작경도'] + link_df['링크끝경도']) / 2
        
        # 도로 폭 컬럼 초기화
        link_df['road_width'] = None
        link_df['matched_road_name'] = None
        link_df['match_distance'] = None
        
        # 도로 데이터의 공간 그리드 인덱스 생성
        print("📍 공간 인덱스 생성 중...")
        road_grid = self.create_spatial_grid(valid_roads, grid_size=0.001)
        print(f"✅ {len(road_grid)}개 그리드 셀 생성")
        
        # 링크별로 매칭 (공간 인덱스 활용)
        processed_links = 0
        for link_idx, link_row in link_df.iterrows():
            processed_links += 1
            
            if processed_links % 1000 == 0:
                progress = (processed_links / len(link_df)) * 100
                print(f"⏳ 링크 매칭 진행률: {processed_links}/{len(link_df)} ({progress:.1f}%)")
            
            # 링크 중점 좌표
            link_lat = link_row['mid_latitude']
            link_lon = link_row['mid_longitude']
            
            if pd.notna(link_lat) and pd.notna(link_lon):
                # 주변 그리드 셀들 검색
                nearby_cells = self.get_nearby_grid_cells(link_lat, link_lon)
                
                best_distance = float('inf')
                best_road = None
                
                # 주변 그리드의 도로들만 검사
                for grid_cell in nearby_cells:
                    if grid_cell in road_grid:
                        for road_idx in road_grid[grid_cell]:
                            road_row = valid_roads.iloc[road_idx]
                            
                            # 거리 계산
                            road_coord = (road_row['latitude'], road_row['longitude'])
                            link_coord = (link_lat, link_lon)
                            distance = self.calculate_distance(road_coord, link_coord)
                            
                            # 최적 매칭 찾기
                            if distance <= max_distance and distance < best_distance:
                                best_distance = distance
                                best_road = road_row
                
                # 매칭 결과 저장
                if best_road is not None:
                    link_df.loc[link_idx, 'road_width'] = best_road['width']
                    link_df.loc[link_idx, 'matched_road_name'] = best_road['road_name']
                    link_df.loc[link_idx, 'match_distance'] = best_distance
        
        # 중점 좌표 컬럼 제거 (임시 컬럼)
        link_df = link_df.drop(['mid_latitude', 'mid_longitude'], axis=1)
        
        matched_count = link_df['road_width'].notna().sum()
        print(f"🎉 링크 매칭 완료: {matched_count}개 링크에 도로 폭 정보 추가")
        
        return link_df
    
    def match_road_width_to_nodes(self, road_coord_df, node_df, max_distance=300):
        """
        공간 인덱싱을 사용한 최적화된 노드 매칭
        
        Args:
            road_coord_df (pd.DataFrame): 좌표가 포함된 도로 폭 데이터
            node_df (pd.DataFrame): 노드 데이터
            max_distance (float): 최대 매칭 거리 (미터)
            
        Returns:
            pd.DataFrame: 도로 폭 정보가 추가된 노드 데이터
        """
        # 좌표 정보가 있는 도로 데이터만 사용
        valid_roads = road_coord_df[road_coord_df['search_success'] == True].copy()
        
        print(f"🚀 최적화된 노드 매칭 시작: {len(valid_roads)}개 도로, {len(node_df)}개 노드")
        
        # 노드 데이터에 도로 폭 컬럼 초기화
        node_df = node_df.copy()
        node_df['road_width'] = None
        node_df['matched_road_name'] = None
        node_df['match_distance'] = None
        
        # 도로 데이터의 공간 그리드 인덱스 생성
        print("📍 노드용 공간 인덱스 생성 중...")
        road_grid = self.create_spatial_grid(valid_roads, grid_size=0.001)
        print(f"✅ {len(road_grid)}개 그리드 셀 생성")
        
        # 노드별로 매칭 (공간 인덱스 활용)
        processed_nodes = 0
        for node_idx, node_row in node_df.iterrows():
            processed_nodes += 1
            
            if processed_nodes % 1000 == 0:
                progress = (processed_nodes / len(node_df)) * 100
                print(f"⏳ 노드 매칭 진행률: {processed_nodes}/{len(node_df)} ({progress:.1f}%)")
            
            # 노드 좌표
            node_lat = node_row['위도']
            node_lon = node_row['경도']
            
            if pd.notna(node_lat) and pd.notna(node_lon):
                # 주변 그리드 셀들 검색
                nearby_cells = self.get_nearby_grid_cells(node_lat, node_lon)
                
                best_distance = float('inf')
                best_road = None
                
                # 주변 그리드의 도로들만 검사
                for grid_cell in nearby_cells:
                    if grid_cell in road_grid:
                        for road_idx in road_grid[grid_cell]:
                            road_row = valid_roads.iloc[road_idx]
                            
                            # 거리 계산
                            road_coord = (road_row['latitude'], road_row['longitude'])
                            node_coord = (node_lat, node_lon)
                            distance = self.calculate_distance(road_coord, node_coord)
                            
                            # 최적 매칭 찾기
                            if distance <= max_distance and distance < best_distance:
                                best_distance = distance
                                best_road = road_row
                
                # 매칭 결과 저장
                if best_road is not None:
                    node_df.loc[node_idx, 'road_width'] = best_road['width']
                    node_df.loc[node_idx, 'matched_road_name'] = best_road['road_name']
                    node_df.loc[node_idx, 'match_distance'] = best_distance
        
        matched_count = node_df['road_width'].notna().sum()
        print(f"🎉 노드 매칭 완료: {matched_count}개 노드에 도로 폭 정보 추가")
        
        return node_df
    
    def save_results(self, enhanced_link_df, enhanced_node_df, road_coord_df, output_dir="output"):
        """
        결과 파일들을 저장
        
        Args:
            enhanced_link_df (pd.DataFrame): 도로 폭 정보가 추가된 링크 데이터
            enhanced_node_df (pd.DataFrame): 도로 폭 정보가 추가된 노드 데이터
            road_coord_df (pd.DataFrame): 좌표 정보가 포함된 도로 데이터
            output_dir (str): 출력 디렉토리
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 결과 파일 저장
        enhanced_link_df.to_csv(f"{output_dir}/busan_link_with_road_width.csv", index=False, encoding='utf-8-sig')
        enhanced_node_df.to_csv(f"{output_dir}/busan_node_with_road_width.csv", index=False, encoding='utf-8-sig')
        road_coord_df.to_csv(f"{output_dir}/road_coordinate_mapping.csv", index=False, encoding='utf-8-sig')
        
        print(f"결과 파일이 {output_dir} 디렉토리에 저장되었습니다:")
        print(f"- busan_link_with_road_width.csv")
        print(f"- busan_node_with_road_width.csv") 
        print(f"- road_coordinate_mapping.csv")
    
    def generate_summary_report(self, enhanced_link_df, enhanced_node_df, road_coord_df):
        """
        매칭 결과 요약 보고서 생성
        
        Args:
            enhanced_link_df (pd.DataFrame): 도로 폭 정보가 추가된 링크 데이터
            enhanced_node_df (pd.DataFrame): 도로 폭 정보가 추가된 노드 데이터
            road_coord_df (pd.DataFrame): 좌표 정보가 포함된 도로 데이터
        """
        # 도로 폭 데이터에서 지역별 분포
        if 'sigungu' in road_coord_df.columns:
            sigungu_dist = road_coord_df['sigungu'].value_counts()
            print(f"\n🗺️ 지역별 도로 분포:")
            for district, count in sigungu_dist.head().items():
                print(f"  {district}: {count}개")
        
        print("\n" + "="*60)
        print("🗺️ 도로 폭 데이터 통합 결과 보고서")
        print("="*60)
        
        # 좌표 검색 성공률
        total_roads = len(road_coord_df)
        successful_coords = road_coord_df['search_success'].sum()
        coord_success_rate = (successful_coords / total_roads) * 100
        
        print(f"\n📍 좌표 검색 결과:")
        print(f"  총 도로 수: {total_roads}개")
        print(f"  좌표 검색 성공: {successful_coords}개")
        print(f"  성공률: {coord_success_rate:.1f}%")
        
        # 링크 매칭 결과
        total_links = len(enhanced_link_df)
        matched_links = enhanced_link_df['road_width'].notna().sum()
        link_match_rate = (matched_links / total_links) * 100
        
        print(f"\n🔗 링크 데이터 매칭 결과:")
        print(f"  총 링크 수: {total_links}개")
        print(f"  도로 폭 매칭 성공: {matched_links}개")
        print(f"  매칭률: {link_match_rate:.1f}%")
        
        # 노드 매칭 결과
        total_nodes = len(enhanced_node_df)
        matched_nodes = enhanced_node_df['road_width'].notna().sum()
        node_match_rate = (matched_nodes / total_nodes) * 100
        
        print(f"\n📍 노드 데이터 매칭 결과:")
        print(f"  총 노드 수: {total_nodes}개")
        print(f"  도로 폭 매칭 성공: {matched_nodes}개") 
        print(f"  매칭률: {node_match_rate:.1f}%")
        
        # 도로 폭 통계
        if matched_links > 0:
            width_stats = enhanced_link_df['road_width'].dropna()
            print(f"\n📊 도로 폭 통계 (링크 기준):")
            print(f"  평균 도로 폭: {width_stats.mean():.1f}m")
            print(f"  최소 도로 폭: {width_stats.min()}m")
            print(f"  최대 도로 폭: {width_stats.max()}m")
            print(f"  중간값: {width_stats.median():.1f}m")
        
        print("\n" + "="*60)


# 메인 실행 함수 수정
def main():
    KAKAO_API_KEY = "c5ebe192c3d007a46ea0deb05f0ea12e"
    CSV_FILE_PATH = "/Users/gamjawon/Downloads/부산광역시_도로구간조서_20250423.csv"
    
    integrator = RoadWidthIntegrator(KAKAO_API_KEY)
    
    print("🚀 도로 폭 데이터 통합 시스템 시작 (중단 재시작 지원)")
    
    # 기존 데이터 로드
    try:
        node_df = pd.read_csv('/Users/gamjawon/busan_node_with_difficulty_all.csv')
        link_df = pd.read_csv('/Users/gamjawon/busan_link_with_difficulty_all.csv')
        print(f"✅ 노드 데이터: {len(node_df)}개")
        print(f"✅ 링크 데이터: {len(link_df)}개")
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return
    
    # 도로 폭 데이터 로드
    try:
        road_width_df = integrator.load_road_width_csv(CSV_FILE_PATH)
        print(f"✅ 도로 폭 데이터: {len(road_width_df)}개")
    except Exception as e:
        print(f"❌ 도로 폭 데이터 로드 실패: {e}")
        return
    
    # 좌표 정보 추가 (중단 재시작 지원)
    print("\n🗺️ 카카오 API로 좌표 정보 검색 중...")
    print("💡 중단된 경우 동일한 명령어로 재시작하면 이어서 처리됩니다.")
    
    try:
        road_coord_df = integrator.create_coordinate_mapping_with_resume(road_width_df)
        integrator.get_api_usage_report()
        
        if len(road_coord_df) > 0:
            print(f"✅ 좌표 매핑 완료: {len(road_coord_df)}개")
            
            # 나머지 처리 계속...
            enhanced_link_df = integrator.match_road_width_to_links(road_coord_df, link_df)
            enhanced_node_df = integrator.match_road_width_to_nodes(road_coord_df, node_df)
            integrator.save_results(enhanced_link_df, enhanced_node_df, road_coord_df)
            integrator.generate_summary_report(enhanced_link_df, enhanced_node_df, road_coord_df)
            
            # 체크포인트 파일 정리
            if os.path.exists(integrator.checkpoint_file):
                os.remove(integrator.checkpoint_file)
                print("🧹 체크포인트 파일 정리 완료")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        print("💾 진행상황이 저장되었습니다. 동일한 명령어로 재시작 가능합니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        integrator.get_api_usage_report()


if __name__ == "__main__":
    main()
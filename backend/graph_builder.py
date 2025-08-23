# graph_builder.py
import pandas as pd
import networkx as nx

def create_graph_from_csv(node_file_path: str, link_file_path: str):
    """
    CSV 파일에서 노드와 링크 데이터를 읽어 networkx 그래프를 생성합니다.
    """
    try:
        diff_node = pd.read_csv(node_file_path, encoding='utf-8-sig')
        diff_link = pd.read_csv(link_file_path, encoding='utf-8-sig')
    except Exception as e:
        raise RuntimeError(f"데이터 로드 오류: {e}")

    # 그래프 G 생성
    G = nx.Graph()
    
    # 노드 ID -> 좌표 맵 생성
    node_coords_map = {row['NODE_ID']: (row['경도'], row['위도']) for _, row in diff_node.iterrows()}
    
    # 링크 추가
    for _, row in diff_link.iterrows():
        f_node = row['시작 노드']
        t_node = row['끝 노드']
        
        # 노드가 모두 유효한지 확인하고 그래프에 추가
        if f_node in node_coords_map and t_node in node_coords_map:
            G.add_edge(f_node, t_node, weight=row['link_difficulty'])
            
    print("그래프 구축 완료!")
    return G, node_coords_map, diff_node, diff_link
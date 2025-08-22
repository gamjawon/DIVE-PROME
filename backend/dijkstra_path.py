import networkx as nx
import heapq
from typing import List, Tuple


def dijkstra_easy_path(graph: nx.Graph, start_node, end_node):
    """
    난이도를 가중치로 사용하여 최적의 경로를 찾는 다익스트라 알고리즘.

    :param graph: networkx 그래프 객체
    :param start_node: 우회 경로의 시작 노드 ID
    :param end_node: 우회 경로의 끝 노드 ID
    :return: (최소 난이도 합계, 경로 노드 리스트)
    """
    # 시작 노드와 끝 노드가 그래프에 없는 경우를 처리
    if start_node not in graph or end_node not in graph:
        print("경고: 시작 노드 또는 끝 노드가 그래프에 존재하지 않습니다.")
        return float('inf'), []

    # 각 노드까지의 최소 난이도를 저장할 딕셔너리 (초기값은 무한대)
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start_node] = 0
    
    # 경로를 역추적하기 위한 딕셔너리
    previous_nodes = {node: None for node in graph.nodes()}
    
    # (난이도, 노드 ID)를 담을 우선순위 큐 (Min-Heap)
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_difficulty, current_node = heapq.heappop(priority_queue)

        # 이미 더 짧은(낮은 난이도) 경로를 찾았다면 스킵
        if current_difficulty > distances[current_node]:
            continue
        
        # 목표 노드에 도달하면 탐색 중단
        if current_node == end_node:
            break

        # 현재 노드와 연결된 이웃 노드 탐색
        for neighbor, data in graph[current_node].items():
            # 엣지의 'weight' 속성을 난이도 가중치로 사용
            link_difficulty = data.get('weight', 0)
            new_difficulty = current_difficulty + link_difficulty
            
            # 더 낮은 난이도의 경로를 찾았다면 업데이트
            if new_difficulty < distances[neighbor]:
                distances[neighbor] = new_difficulty
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (new_difficulty, neighbor))

    # 경로 역추적 및 반환
    path = []
    current = end_node
    while current is not None:
        path.insert(0, current)
        current = previous_nodes.get(current)
    
    if path and path[0] == start_node:
        return distances[end_node], path
    else:
        # 경로를 찾지 못한 경우
        return float('inf'), []
# DIVE-PROME (Easy Navi)

고령자와 초보운전자를 위한 쉽고 안전한 네비게이션

## 📖 개요

안전하고 쉬운 길안내를 목표로 한 내비게이션 앱입니다. 카카오 길찾기 API와 정적 데이터 기반 난이도·혼잡도 요소를 결합해 “쉬운(EASY) 경로”를 제공합니다.

## ✨ 주요 기능

- 🗺️ Kakao Map 렌더링 및 현재 위치 연동(권한 처리 포함)
- 🔎 장소 검색(카카오 로컬 API)과 현위치 선택 UX
- 🧭 3가지 경로 옵션 제공: RECOMMEND / MAIN_ROAD / EASY
- 📊 경로 메타 정보 표출: 소요시간, 거리, 유턴/차선변경/급경사로 수

## 🛠 기술 스택

- **Frontend**: Flutter (Dart 3.9), Kakao Map SDK, Geolocator, Riverpod, Freezed, http
- **Backend**: FastAPI, NetworkX, NumPy/SciPy(cKDTree), Pandas, Pydantic

## 🧩 아키텍처 요약

- **Frontend**: MVVM + Clean Architecture(data/domain/presentation), Riverpod 기반 DI/상태관리
- **Backend**: FastAPI + 그래프 기반 후처리(NetworkX), 난이도/혼잡도/차선변경 패널티, KD-Tree 근접 노드 매핑, 폴리라인 평활화(표시용)

## 📦 프로젝트 구조

```
DIVE-PROME/
	backend/
		app.py
		graph_builder.py
		difficulty_scorer.py
		cal_lane_chng.py
		dijkstra_path.py
		utils.py

	frontend/
		lib/
			data/
				datasources/
				models/
				repositories/
				utils/
			domain/
				entities/
				enums/
				repositories/
				value_objects/
			presentation/
				views/
				viewmodels/
				states/
				utils/
			main.dart
		assets/
			icons/, images/
        ...
```

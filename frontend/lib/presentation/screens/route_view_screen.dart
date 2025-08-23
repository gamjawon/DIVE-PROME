import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/providers/route_provider.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

class RouteViewScreen extends ConsumerStatefulWidget {
  const RouteViewScreen({super.key});

  @override
  ConsumerState<RouteViewScreen> createState() => _RouteViewScreenState();
}

class _RouteViewScreenState extends ConsumerState<RouteViewScreen> {
  KakaoMapController? _mapController;
  List<LatLng> _routePoints = [];

  @override
  Widget build(BuildContext context) {
    final routeState = ref.watch(routeProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('경로 보기'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      body: routeState.when(
        data: (routeResponse) {
          if (routeResponse == null) {
            return const Center(child: Text('경로 데이터가 없습니다.'));
          }

          // 경로 좌표를 LatLng 리스트로 변환
          _routePoints = routeResponse.pathPoints
              .map(
                (point) => LatLng(point[1], point[0]),
              ) // [경도, 위도] -> LatLng(위도, 경도)
              .toList();

          if (_routePoints.isEmpty) {
            return const Center(child: Text('경로 좌표가 없습니다.'));
          }

          return _buildMap();
        },
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, stack) => Center(child: Text('오류: $error')),
      ),
    );
  }

  Widget _buildMap() {
    // 경로의 중심점 계산
    final centerLat =
        _routePoints.map((p) => p.latitude).reduce((a, b) => a + b) /
        _routePoints.length;
    final centerLng =
        _routePoints.map((p) => p.longitude).reduce((a, b) => a + b) /
        _routePoints.length;

    return KakaoMap(
      option: KakaoMapOption(
        position: LatLng(centerLat, centerLng),
        zoomLevel: 15,
        mapType: MapType.normal,
      ),
      onMapReady: (controller) {
        _mapController = controller;
        _drawRoute();
        // 지도가 완전히 로드된 후 카메라 조정
        Future.delayed(const Duration(milliseconds: 300), () {
          _adjustCamera();
        });
      },
    );
  }

  void _drawRoute() {
    if (_mapController == null || _routePoints.isEmpty) return;

    // 경로 그리기
    _mapController!.routeLayer.addRoute(
      _routePoints,
      RouteStyle(
        const Color(0xFFFF5930), // 경로 색상 (주황색)
        8.0, // 경로 굵기
      ),
    );

    // 시작점 POI 추가
    _mapController!.labelLayer.addPoi(
      _routePoints.first,
      style: PoiStyle(),
      text: '출발',
    );

    // 도착점 POI 추가
    _mapController!.labelLayer.addPoi(
      _routePoints.last,
      style: PoiStyle(),
      text: '도착',
    );
  }

  void _adjustCamera() {
    if (_mapController == null || _routePoints.isEmpty) return;

    // 경로의 경계 계산 (모든 경로 포인트를 고려)
    double minLat = _routePoints.first.latitude;
    double maxLat = _routePoints.first.latitude;
    double minLng = _routePoints.first.longitude;
    double maxLng = _routePoints.first.longitude;

    for (final point in _routePoints) {
      if (point.latitude < minLat) minLat = point.latitude;
      if (point.latitude > maxLat) maxLat = point.latitude;
      if (point.longitude < minLng) minLng = point.longitude;
      if (point.longitude > maxLng) maxLng = point.longitude;
    }

    // 경계에 적절한 여백 추가 (위도/경도 기준)
    final latRange = maxLat - minLat;
    final lngRange = maxLng - minLng;

    // 여백을 위도/경도 단위로 계산
    // 최소 여백 설정 (너무 작은 경로를 위해)
    double latPadding = latRange * 0.25; // 25% 여백
    double lngPadding = lngRange * 0.25; // 25% 여백

    // 최소 여백 보장 (약 200m 정도)
    const minPaddingDegrees = 0.002; // 대략 200m
    if (latPadding < minPaddingDegrees) latPadding = minPaddingDegrees;
    if (lngPadding < minPaddingDegrees) lngPadding = minPaddingDegrees;

    // 최대 여백 제한 (너무 큰 여백 방지)
    const maxPaddingDegrees = 0.01; // 대략 1km
    if (latPadding > maxPaddingDegrees) latPadding = maxPaddingDegrees;
    if (lngPadding > maxPaddingDegrees) lngPadding = maxPaddingDegrees;

    // 여백을 포함한 최종 범위
    final finalMinLat = minLat - latPadding;
    final finalMaxLat = maxLat + latPadding;
    final finalMinLng = minLng - lngPadding;
    final finalMaxLng = maxLng + lngPadding;

    // 최종 범위 계산
    final totalLatRange = finalMaxLat - finalMinLat;
    final totalLngRange = finalMaxLng - finalMinLng;

    // 중심점
    final finalCenterLat = (finalMinLat + finalMaxLat) / 2;
    final finalCenterLng = (finalMinLng + finalMaxLng) / 2;

    // 줌 레벨을 범위에 따라 계산 (더 보수적으로)
    // 위도/경도 단위를 고려한 실용적 접근
    final maxDimension = totalLatRange > totalLngRange
        ? totalLatRange
        : totalLngRange;

    int zoomLevel;
    if (maxDimension > 4.0) {
      // 극대 범위 (~450km+) - 한반도 전체
      zoomLevel = 5;
    } else if (maxDimension > 2.0) {
      // 매우 매우 큰 범위 (~220km) - 서울-부산급
      zoomLevel = 6;
    } else if (maxDimension > 1.0) {
      // 대형 범위 (~110km) - 서울-대전급
      zoomLevel = 7;
    } else if (maxDimension > 0.5) {
      // 큰 범위 (~55km) - 수도권 전체급
      zoomLevel = 8;
    } else if (maxDimension > 0.2) {
      // 매우 큰 범위 (~22km) - 서울시 전체급
      zoomLevel = 9;
    } else if (maxDimension > 0.15) {
      // 큰 범위 (~17km)
      zoomLevel = 10;
    } else if (maxDimension > 0.1) {
      // 중큰 범위 (~11km)
      zoomLevel = 11;
    } else if (maxDimension > 0.08) {
      // 꽤 큰 범위 (~9km)
      zoomLevel = 12;
    } else if (maxDimension > 0.04) {
      // 중간 큰 범위 (~4.5km)
      zoomLevel = 13;
    } else if (maxDimension > 0.02) {
      // 중간 범위 (~2.2km)
      zoomLevel = 14;
    } else if (maxDimension > 0.01) {
      // 보통 범위 (~1.1km)
      zoomLevel = 15;
    } else if (maxDimension > 0.006) {
      // 작은 범위 (~650m)
      zoomLevel = 16;
    } else if (maxDimension > 0.003) {
      // 매우 작은 범위 (~330m)
      zoomLevel = 17;
    } else {
      // 극히 작은 범위 (~330m 미만)
      zoomLevel = 18;
    }

    print('=== 카메라 조정 정보 ===');
    print(
      '경로 범위: 위도 ${latRange.toStringAsFixed(6)}, 경도 ${lngRange.toStringAsFixed(6)}',
    );
    print(
      '여백: 위도 ${latPadding.toStringAsFixed(6)}, 경도 ${lngPadding.toStringAsFixed(6)}',
    );
    print(
      '최종 범위: 위도 ${totalLatRange.toStringAsFixed(6)}, 경도 ${totalLngRange.toStringAsFixed(6)}',
    );
    print('최대 차원: ${maxDimension.toStringAsFixed(6)}');
    print(
      '중심점: (${finalCenterLat.toStringAsFixed(6)}, ${finalCenterLng.toStringAsFixed(6)})',
    );
    print('줌 레벨: $zoomLevel');

    // 카메라 이동
    _mapController!.moveCamera(
      CameraUpdate.newCenterPosition(LatLng(finalCenterLat, finalCenterLng)),
    );

    // 줌 레벨 조정
    Future.delayed(const Duration(milliseconds: 300), () {
      _mapController?.moveCamera(CameraUpdate.zoomTo(zoomLevel));
    });
  }
}

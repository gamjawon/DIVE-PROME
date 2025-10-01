import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/models/route_model.dart';
import 'package:frontend/presentation/states/route_state.dart';
import 'package:frontend/presentation/utils/palette.dart';
import 'package:frontend/presentation/viewmodels/route_viewmodel.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

class RouteMap extends ConsumerStatefulWidget {
  const RouteMap({super.key});

  @override
  ConsumerState<ConsumerStatefulWidget> createState() => _RouteMapState();
}

class _RouteMapState extends ConsumerState<RouteMap> {
  KakaoMapController? _mapController;
  RouteOption? _lastSelectedOption;
  List<RouteInfo>? _lastRouteList;

  void _drawAllRoutes(RouteState routeState) {
    if (_mapController == null || routeState.routeList == null) return;

    final routeList = routeState.routeList!;

    // 먼저 비활성화된 경로들을 그리기 (아래 레이어)
    for (final route in routeList) {
      if (route.option != routeState.selectedOption) {
        final routePoints = route.pathPoints
            .map((point) => LatLng(point[1], point[0]))
            .toList();

        if (routePoints.isNotEmpty) {
          _mapController!.routeLayer.addRoute(
            routePoints,
            RouteStyle(
              Palette.inactiveRouteColor,
              20,
              strokeColor: Colors.white,
              strokeWidth: 4,
            ),
          );
        }
      }
    }

    // 그 다음 활성화된 경로를 그리기 (위 레이어)
    final selectedRoute = routeList.firstWhere(
      (route) => route.option == routeState.selectedOption,
      orElse: () => routeList.first,
    );

    final selectedRoutePoints = selectedRoute.pathPoints
        .map((point) => LatLng(point[1], point[0]))
        .toList();

    if (selectedRoutePoints.isNotEmpty) {
      _mapController!.routeLayer.addRoute(
        selectedRoutePoints,
        RouteStyle(
          Palette.routeColors[routeState.selectedOption]!,
          20,
          strokeColor: Colors.white,
          strokeWidth: 4,
        ),
      );

      // 시작점과 도착점 POI 추가
      final startPoint = LatLng(
        selectedRoute.pathPoints.first[1],
        selectedRoute.pathPoints.first[0],
      );
      final endPoint = LatLng(
        selectedRoute.pathPoints.last[1],
        selectedRoute.pathPoints.last[0],
      );

      // 시작점 POI
      _mapController!.labelLayer.addPoi(
        startPoint,
        style: PoiStyle(
          icon: KImage.fromAsset('assets/icons/my_location.png', 40, 40),
        ),
      );

      // 도착점 POI
      _mapController!.labelLayer.addPoi(
        endPoint,
        style: PoiStyle(icon: KImage.fromAsset('assets/icons/pin.png', 27, 36)),
      );
    }
  }

  void _adjustCamera(RouteState routeState) {
    if (_mapController == null || routeState.routeList == null) return;

    final routeList = routeState.routeList!;

    // 모든 경로 포인트를 고려하여 경계 계산
    final allPoints = <LatLng>[];
    for (final route in routeList) {
      allPoints.addAll(
        route.pathPoints.map((point) => LatLng(point[1], point[0])),
      );
    }

    if (allPoints.isEmpty) return;

    double minLat = allPoints.first.latitude;
    double maxLat = allPoints.first.latitude;
    double minLng = allPoints.first.longitude;
    double maxLng = allPoints.first.longitude;

    for (final point in allPoints) {
      if (point.latitude < minLat) minLat = point.latitude;
      if (point.latitude > maxLat) maxLat = point.latitude;
      if (point.longitude < minLng) minLng = point.longitude;
      if (point.longitude > maxLng) maxLng = point.longitude;
    }

    // 경계에 여백 추가
    final latRange = maxLat - minLat;
    final lngRange = maxLng - minLng;

    double latPadding = latRange * 0.25;
    double lngPadding = lngRange * 0.25;

    const minPaddingDegrees = 0.002;
    if (latPadding < minPaddingDegrees) latPadding = minPaddingDegrees;
    if (lngPadding < minPaddingDegrees) lngPadding = minPaddingDegrees;

    const maxPaddingDegrees = 0.01;
    if (latPadding > maxPaddingDegrees) latPadding = maxPaddingDegrees;
    if (lngPadding > maxPaddingDegrees) lngPadding = maxPaddingDegrees;

    final finalMinLat = minLat - latPadding;
    final finalMaxLat = maxLat + latPadding;
    final finalMinLng = minLng - lngPadding;
    final finalMaxLng = maxLng + lngPadding;

    final totalLatRange = finalMaxLat - finalMinLat;
    final totalLngRange = finalMaxLng - finalMinLng;

    final finalCenterLat = (finalMinLat + finalMaxLat) / 2;
    final finalCenterLng = (finalMinLng + finalMaxLng) / 2;

    final maxDimension = totalLatRange > totalLngRange
        ? totalLatRange
        : totalLngRange;

    int zoomLevel;
    if (maxDimension > 4.0) {
      zoomLevel = 5;
    } else if (maxDimension > 2.0) {
      zoomLevel = 6;
    } else if (maxDimension > 1.0) {
      zoomLevel = 7;
    } else if (maxDimension > 0.5) {
      zoomLevel = 8;
    } else if (maxDimension > 0.2) {
      zoomLevel = 9;
    } else if (maxDimension > 0.15) {
      zoomLevel = 10;
    } else if (maxDimension > 0.1) {
      zoomLevel = 11;
    } else if (maxDimension > 0.08) {
      zoomLevel = 12;
    } else if (maxDimension > 0.04) {
      zoomLevel = 13;
    } else if (maxDimension > 0.02) {
      zoomLevel = 14;
    } else if (maxDimension > 0.01) {
      zoomLevel = 15;
    } else if (maxDimension > 0.006) {
      zoomLevel = 16;
    } else if (maxDimension > 0.003) {
      zoomLevel = 17;
    } else {
      zoomLevel = 18;
    }

    _mapController!.moveCamera(
      CameraUpdate.newCenterPosition(LatLng(finalCenterLat, finalCenterLng)),
    );

    Future.delayed(const Duration(milliseconds: 300), () {
      _mapController?.moveCamera(CameraUpdate.zoomTo(zoomLevel));
    });
  }

  @override
  Widget build(BuildContext context) {
    final routeStateAsync = ref.watch(routeViewmodelProvider);

    return routeStateAsync.when(
      data: (routeState) => _buildMapWidget(context, routeState),
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (_, __) => const Center(child: Text('경로를 불러올 수 없습니다.')),
    );
  }

  Widget _buildMapWidget(BuildContext context, RouteState routeState) {
    if (routeState.routeList == null || routeState.routeList!.isEmpty) {
      return const Center(child: Text('경로 데이터가 없습니다.'));
    }

    // 상태 변화 감지 및 경로 다시 그리기
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_mapController != null &&
          (routeState.selectedOption != _lastSelectedOption ||
              routeState.routeList != _lastRouteList)) {
        _lastSelectedOption = routeState.selectedOption;
        _lastRouteList = routeState.routeList;

        _drawAllRoutes(routeState);
        // _adjustCamera(routeState);
      }
    });

    // 첫 번째 경로의 중심점으로 지도 초기화
    final routeList = routeState.routeList!;
    final firstRoute = routeList.first;
    if (firstRoute.pathPoints.isEmpty) {
      return const Center(child: Text('경로 좌표가 없습니다.'));
    }

    final centerLat =
        firstRoute.pathPoints.map((p) => p[1]).reduce((a, b) => a + b) /
        firstRoute.pathPoints.length;
    final centerLng =
        firstRoute.pathPoints.map((p) => p[0]).reduce((a, b) => a + b) /
        firstRoute.pathPoints.length;

    return KakaoMap(
      option: KakaoMapOption(
        position: LatLng(centerLat, centerLng),
        zoomLevel: 15,
        mapType: MapType.normal,
      ),
      onMapReady: (controller) {
        _mapController = controller;
        _drawAllRoutes(routeState);
        _adjustCamera(routeState);
      },
    );
  }
}

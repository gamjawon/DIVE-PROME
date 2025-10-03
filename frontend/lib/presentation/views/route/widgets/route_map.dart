import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/models/route_model.dart';
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

  void _drawAllRoutes() {
    final routeStateAsync = ref.read(routeViewmodelProvider);

    routeStateAsync.whenData((routeState) {
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
          style: PoiStyle(
            icon: KImage.fromAsset('assets/icons/pin.png', 27, 36),
          ),
        );
      }
    });
  }

  void _adjustCamera() {
    final routeStateAsync = ref.read(routeViewmodelProvider);

    routeStateAsync.whenData((routeState) {
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

      final finalMinLat = minLat - latPadding;
      final finalMaxLat = maxLat + latPadding;
      final finalMinLng = minLng - lngPadding;
      final finalMaxLng = maxLng + lngPadding;

      _mapController!.moveCamera(
        CameraUpdate.fitMapPoints([
          LatLng(finalMaxLat, finalMaxLng),
          LatLng(finalMinLat, finalMinLng),
        ]),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final routeStateAsync = ref.watch(routeViewmodelProvider);

    return routeStateAsync.when(
      data: (routeState) {
        if (routeState.routeList == null || routeState.routeList!.isEmpty) {
          return const Center(child: Text('경로 데이터가 없습니다.'));
        }

        // 상태 변화 감지 및 경로 다시 그리기
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (_mapController != null &&
              (routeState.selectedOption != _lastSelectedOption)) {
            _lastSelectedOption = routeState.selectedOption;

            _drawAllRoutes();
            // _adjustCamera();
          }
        });

        return KakaoMap(
          onMapReady: (controller) {
            _mapController = controller;
            _drawAllRoutes();
            _adjustCamera();
          },
        );
      },
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (_, __) => const Center(child: Text('경로를 불러올 수 없습니다.')),
    );
  }
}

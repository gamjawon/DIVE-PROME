import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/svg.dart';
import 'package:frontend/models/route_response_model.dart';
import 'package:frontend/presentation/screens/navi_screen.dart';
import 'package:frontend/providers/route_provider.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

class RouteViewScreen extends ConsumerStatefulWidget {
  const RouteViewScreen({super.key});

  @override
  ConsumerState<RouteViewScreen> createState() => _RouteViewScreenState();
}

class _RouteViewScreenState extends ConsumerState<RouteViewScreen> {
  KakaoMapController? _mapController;
  RouteOption _selectedOption = RouteOption.easy;
  RouteResponse? _routeResponse;

  // 경로별 색상 정의
  static const Map<RouteOption, Color> _routeColors = {
    RouteOption.easy: Color(0xFFFF792B), // 주황색
    RouteOption.navi: Color(0xFF4285F4), // 파란색
    RouteOption.wide: Color(0xFF34A853), // 초록색
  };
  static const Color _inactiveRouteColor = Color(0xFFD1D5DB);

  @override
  Widget build(BuildContext context) {
    final routeState = ref.watch(routeProvider);

    return Scaffold(
      body: Stack(
        children: [
          routeState.when(
            data: (routeResponse) {
              if (routeResponse == null || routeResponse.routes.isEmpty) {
                return const Center(child: Text('경로 데이터가 없습니다.'));
              }

              _routeResponse = routeResponse;
              return _buildMap();
            },
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (error, stack) => Center(child: Text('오류: $error')),
          ),
          _buildTopBar(),
          // 하단 여백을 채우는 컨테이너
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            height: 20,
            child: Container(color: Colors.white),
          ),
          _buildBottomContainer(),
        ],
      ),
    );
  }

  Widget _buildTopBar() {
    return Positioned(
      top: 70,
      left: 40,
      right: 40,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: SvgPicture.asset(
              'assets/icons/back.svg',
              width: 30,
              height: 30,
            ),
          ),
          Container(
            width: 300,
            height: 60,
            decoration: ShapeDecoration(
              color: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15),
              ),
              shadows: [
                BoxShadow(
                  color: Color(0x3F9F9F9F),
                  blurRadius: 3.84,
                  offset: Offset(-0.96, 0),
                  spreadRadius: 0,
                ),
                BoxShadow(
                  color: Color(0x3F787878),
                  blurRadius: 3.84,
                  offset: Offset(0.96, 0.96),
                  spreadRadius: 0,
                ),
              ],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '부산역',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: const Color(0xFF374151),
                    fontSize: 20,
                    fontFamily: 'Pretendard',
                    fontWeight: FontWeight.w500,
                    height: 1.29,
                  ),
                ),
                Container(
                  margin: EdgeInsets.symmetric(horizontal: 10),
                  child: SvgPicture.asset(
                    'assets/icons/arrow.svg',
                    width: 16,
                    height: 16,
                  ),
                ),
                Text(
                  '서면교차로',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: const Color(0xFF374151),
                    fontSize: 20,
                    fontFamily: 'Pretendard',
                    fontWeight: FontWeight.w500,
                    height: 1.29,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomContainer() {
    if (_routeResponse == null) return SizedBox.shrink();

    final selectedRoute = _routeResponse!.routes.firstWhere(
      (route) => route.option == _selectedOption,
      orElse: () => _routeResponse!.routes.first,
    );

    return Positioned(
      bottom: 20,
      left: 0,
      right: 0,
      child: Container(
        decoration: ShapeDecoration(
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.only(
              topLeft: Radius.circular(23.06),
              topRight: Radius.circular(23.06),
            ),
          ),
        ),
        clipBehavior: Clip.antiAlias,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // 경로 옵션 탭
            Container(
              height: 76,
              child: Row(
                children: RouteOption.values.map((option) {
                  final isSelected = option == _selectedOption;
                  final optionColor = _routeColors[option]!;
                  return Expanded(
                    child: GestureDetector(
                      onTap: () => _selectRouteOption(option),
                      child: Container(
                        decoration: BoxDecoration(
                          color: isSelected
                              ? optionColor.withOpacity(0.1)
                              : Colors.white,
                          borderRadius: BorderRadius.only(
                            topLeft: Radius.circular(
                              option == RouteOption.easy ? 23.06 : 0,
                            ),
                            topRight: Radius.circular(
                              option == RouteOption.wide ? 23.06 : 0,
                            ),
                          ),
                        ),
                        child: Center(
                          child: Text(
                            option.displayName,
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: isSelected
                                  ? FontWeight.w600
                                  : FontWeight.w400,
                              color: isSelected
                                  ? optionColor
                                  : Color(0xFF6B7280),
                            ),
                          ),
                        ),
                      ),
                    ),
                  );
                }).toList(),
              ),
            ),
            // 선택된 경로 정보
            Container(
              padding: const EdgeInsets.fromLTRB(32, 16, 32, 36),
              decoration: BoxDecoration(color: Colors.white),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: 8),
                  // 시간과 거리, 안내시작 버튼
                  Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // 시간과 거리
                            Row(
                              children: [
                                Text(
                                  '${selectedRoute.duration}분',
                                  style: TextStyle(
                                    fontSize: 40,
                                    fontWeight: FontWeight.w700,
                                    color: Color(0xFF111827),
                                    height: 1.0,
                                  ),
                                ),
                                Container(
                                  margin: EdgeInsets.symmetric(horizontal: 16),
                                  height: 30,
                                  decoration: ShapeDecoration(
                                    shape: RoundedRectangleBorder(
                                      side: BorderSide(
                                        width: 1,
                                        strokeAlign:
                                            BorderSide.strokeAlignCenter,
                                        color: const Color(0xFFD1D5DB),
                                      ),
                                    ),
                                  ),
                                ),
                                Text(
                                  '${selectedRoute.distance.toStringAsFixed(0)}km',
                                  style: TextStyle(
                                    fontSize: 24,
                                    fontWeight: FontWeight.w400,
                                    color: Color(0xFF6B7280),
                                  ),
                                ),
                              ],
                            ),
                            SizedBox(height: 24),
                            // 경로 특성 태그
                            Row(
                              children: [
                                _buildRouteTag(
                                  '완만함',
                                  selectedRoute.steepRoads <= 1,
                                ),
                                SizedBox(width: 8),
                                _buildRouteTag(
                                  '넓은 폭',
                                  selectedRoute.option == RouteOption.wide,
                                ),
                                SizedBox(width: 8),
                                _buildRouteTag(
                                  '낮은 경사',
                                  selectedRoute.steepRoads == 0,
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (_) => NaviScreen()),
                          );
                        },
                        child: Column(
                          children: [
                            Container(
                              width: 80,
                              height: 80,
                              padding: EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 12,
                              ),
                              decoration: BoxDecoration(
                                color: _routeColors[_selectedOption]!,
                                borderRadius: BorderRadius.circular(100),
                              ),
                              child: SvgPicture.asset(
                                'assets/icons/start_navi.svg',
                                width: 30,
                                height: 30,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              '안내시작',
                              style: TextStyle(
                                color: _routeColors[_selectedOption]!,
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  Container(
                    margin: EdgeInsets.symmetric(vertical: 32),
                    width: MediaQuery.sizeOf(context).width - 32,
                    height: 1,
                    color: const Color(0xFFC9C9C9),
                  ),
                  // 상세 통계
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _buildStatItem(
                        '차선 변경',
                        '${selectedRoute.laneChanges}회',
                        _routeColors[_selectedOption]!,
                      ),
                      _buildStatItem(
                        'U턴 횟수',
                        selectedRoute.hasUturn ? '있음' : '없음',
                        _routeColors[_selectedOption]!,
                      ),
                      _buildStatItem(
                        '급경사 횟수',
                        '${selectedRoute.steepRoads}회',
                        _routeColors[_selectedOption]!,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRouteTag(String label, bool isHighlighted) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Color(0xFFEFEFF0),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Color(0xFFD1D5DB), width: 1),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w500,
          color: Color(0xFF6B7280),
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, Color routeColor) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 20,
            color: Colors.black45,
            fontWeight: FontWeight.w500,
          ),
        ),
        SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 20,
            color: routeColor,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  void _selectRouteOption(RouteOption option) {
    setState(() {
      _selectedOption = option;
    });
    // 지도 다시 그리기
    Future.delayed(const Duration(milliseconds: 100), () {
      _drawAllRoutes();
    });
  }

  Widget _buildMap() {
    if (_routeResponse == null || _routeResponse!.routes.isEmpty) {
      return const Center(child: Text('경로 데이터가 없습니다.'));
    }

    // 첫 번째 경로의 중심점으로 지도 초기화
    final firstRoute = _routeResponse!.routes.first;
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
        _drawAllRoutes();
        _adjustCamera();
      },
    );
  }

  void _drawAllRoutes() {
    if (_mapController == null || _routeResponse == null) return;

    // 먼저 비활성화된 경로들을 그리기 (아래 레이어)
    for (final route in _routeResponse!.routes) {
      if (route.option != _selectedOption) {
        final routePoints = route.pathPoints
            .map((point) => LatLng(point[1], point[0]))
            .toList();

        if (routePoints.isNotEmpty) {
          _mapController!.routeLayer.addRoute(
            routePoints,
            RouteStyle(
              _inactiveRouteColor,
              20,
              strokeColor: Colors.white,
              strokeWidth: 4,
            ),
          );
        }
      }
    }

    // 그 다음 활성화된 경로를 그리기 (위 레이어)
    final selectedRoute = _routeResponse!.routes.firstWhere(
      (route) => route.option == _selectedOption,
      orElse: () => _routeResponse!.routes.first,
    );

    final selectedRoutePoints = selectedRoute.pathPoints
        .map((point) => LatLng(point[1], point[0]))
        .toList();

    if (selectedRoutePoints.isNotEmpty) {
      _mapController!.routeLayer.addRoute(
        selectedRoutePoints,
        RouteStyle(
          _routeColors[_selectedOption]!,
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

  void _adjustCamera() {
    if (_mapController == null || _routeResponse == null) return;

    // 모든 경로 포인트를 고려하여 경계 계산
    final allPoints = <LatLng>[];
    for (final route in _routeResponse!.routes) {
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
}

import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/route_request_model.dart';
import '../models/route_response_model.dart';

class RouteApiService {
  static const String baseUrl = 'http://10.0.2.2:8000';

  static Future<RouteResponse> getRoute(RouteRequest request) async {
    try {
      print('API 요청: ${request.toJson()}');

      final response = await http.post(
        Uri.parse('$baseUrl/find-path'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(request.toJson()),
      );

      print('API 응답 상태: ${response.statusCode}');
      print('API 응답 내용: ${response.body}');

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);

        // API에서 받은 경로 데이터 추출
        final apiPathPoints =
            (responseData['path_points'] as List<dynamic>?)
                ?.map(
                  (point) => (point as List<dynamic>)
                      .map((coord) => (coord as num).toDouble())
                      .toList(),
                )
                .toList()
                .cast<List<double>>() ??
            [];

        // 실제 경로 데이터를 기반으로 3가지 변형 경로 생성
        return _createThreeRoutesFromApi(apiPathPoints);
      } else {
        throw Exception('Failed to get route: ${response.statusCode}');
      }
    } catch (e) {
      print('API 에러: $e');
      // API 실패 시 하드코딩된 값 반환
      return _getHardcodedRouteResponse();
    }
  }

  static RouteResponse _createThreeRoutesFromApi(
    List<List<double>> apiPathPoints,
  ) {
    if (apiPathPoints.isEmpty) {
      return _getHardcodedRouteResponse();
    }

    return RouteResponse(
      routes: [
        // 쉬운 길 추천 - 원본 경로 사용
        RouteInfo(
          option: RouteOption.easy,
          pathPoints: apiPathPoints,
          distance: 22.0,
          duration: 30,
          laneChanges: 3,
          hasUturn: false,
          steepRoads: 1,
        ),
        // 내비 추천 - 약간 변형된 경로
        RouteInfo(
          option: RouteOption.navi,
          pathPoints: _createVariantRoute(apiPathPoints, 1),
          distance: 20.5,
          duration: 28,
          laneChanges: 4,
          hasUturn: true,
          steepRoads: 1,
        ),
        // 큰길 우선 - 더 변형된 경로
        RouteInfo(
          option: RouteOption.wide,
          pathPoints: _createVariantRoute(apiPathPoints, 2),
          distance: 19.8,
          duration: 25,
          laneChanges: 6,
          hasUturn: false,
          steepRoads: 2,
        ),
      ],
    );
  }

  static RouteResponse _getHardcodedRouteResponse() {
    // 하드코딩된 경로 데이터 (부산 지역 예시)
    final baseRoute = [
      [129.0756, 35.1171], // 부산역
      [129.0750, 35.1180],
      [129.0745, 35.1190],
      [129.0740, 35.1200],
      [129.0735, 35.1210],
      [129.0730, 35.1220],
      [129.0725, 35.1230],
      [129.0720, 35.1240], // 서면교차로 인근
    ];

    return RouteResponse(
      routes: [
        RouteInfo(
          option: RouteOption.easy,
          pathPoints: baseRoute,
          distance: 22.0,
          duration: 30,
          laneChanges: 3,
          hasUturn: false,
          steepRoads: 1,
        ),
        RouteInfo(
          option: RouteOption.navi,
          pathPoints: _createVariantRoute(baseRoute, 1),
          distance: 20.5,
          duration: 28,
          laneChanges: 4,
          hasUturn: true,
          steepRoads: 1,
        ),
        RouteInfo(
          option: RouteOption.wide,
          pathPoints: _createVariantRoute(baseRoute, 2),
          distance: 19.8,
          duration: 25,
          laneChanges: 6,
          hasUturn: false,
          steepRoads: 2,
        ),
      ],
    );
  }

  static List<List<double>> _createVariantRoute(
    List<List<double>> baseRoute,
    int variant,
  ) {
    if (baseRoute.isEmpty) return baseRoute;

    return baseRoute.asMap().entries.map((entry) {
      final index = entry.key;
      final point = entry.value;

      // 시작점과 끝점은 변경하지 않음
      if (index == 0 || index == baseRoute.length - 1) {
        return point;
      }

      // 각 경로별로 약간씩 다른 좌표를 생성
      double offsetLng = 0.0;
      double offsetLat = 0.0;

      // 경로 중간 지점들에만 오프셋 적용
      final offsetFactor = (index / baseRoute.length) * 0.001;

      switch (variant) {
        case 1: // 내비 추천
          offsetLng = offsetFactor;
          offsetLat = offsetFactor * 0.5;
          break;
        case 2: // 큰길 우선
          offsetLng = offsetFactor * 1.5;
          offsetLat = -offsetFactor * 0.5;
          break;
        default: // 쉬운 길
          offsetLng = 0.0;
          offsetLat = 0.0;
          break;
      }

      return [
        point[0] + offsetLng, // 경도
        point[1] + offsetLat, // 위도
      ];
    }).toList();
  }
}

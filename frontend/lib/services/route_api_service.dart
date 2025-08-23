import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/route_request_model.dart';
import '../models/route_response_model.dart';

class RouteApiService {
  // 플랫폼별 로컬 개발 URL
  static String get baseUrl {
    if (Platform.isAndroid) {
      return 'http://10.0.2.2:8000'; // 안드로이드 에뮬레이터용
    } else if (Platform.isIOS) {
      return 'http://localhost:8000'; // iOS 시뮬레이터용
    } else {
      return 'http://localhost:8000'; // 기타 플랫폼 (웹, 데스크톱)
    }
  }

  static Future<RouteResponse> getRoute(RouteRequest request) async {
    final url = '$baseUrl/find-path';
    try {
      print('API 요청 URL: $url');
      print('API 요청 데이터: ${request.toJson()}');

      final response = await http
          .post(
            Uri.parse(url),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(request.toJson()),
          )
          .timeout(
            const Duration(seconds: 30), // 30초 타임아웃
            onTimeout: () {
              throw Exception('API 호출 타임아웃 (30초)');
            },
          );

      print('API 응답 상태: ${response.statusCode}');
      if (response.statusCode == 200) {
        print('API 응답 성공: ${response.body.length} bytes');
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        return RouteResponse.fromJson(responseData);
      } else {
        print('API 응답 실패: ${response.body}');
        throw Exception(
          'Failed to get route: ${response.statusCode} - ${response.body}',
        );
      }
    } catch (e) {
      print('API 에러 상세: $e');
      print('사용 중인 URL: $url');
      print('플랫폼: ${Platform.operatingSystem}');
      // API 실패 시 더미 데이터 반환
      return _getDummyRouteResponse();
    }
  }

  // API 실패 시 사용할 더미 데이터
  static RouteResponse _getDummyRouteResponse() {
    // 부산 지역 더미 경로 데이터
    final dummyPathPoints = [
      [129.0756, 35.1171], // 부산역
      [129.0750, 35.1180],
      [129.0745, 35.1190],
      [129.0740, 35.1200],
      [129.0735, 35.1210],
      [129.0730, 35.1220],
      [129.0725, 35.1230],
      [129.0720, 35.1240], // 서면교차로 인근
    ];

    final dummyDisplayPoints = [
      [129.0756, 35.1171], // 부산역
      [129.0740, 35.1200],
      [129.0720, 35.1240], // 서면교차로 인근
    ];

    return RouteResponse(
      routes: {
        'EASY': RouteInfo(
          label: 'EASY',
          pathPoints: dummyPathPoints,
          displayPathPoints: dummyDisplayPoints,
          distanceM: 2200.0,
          durationSec: 1800, // 30분
          laneChanges: 3,
          uTurns: 0,
        ),
        'RECOMMEND': RouteInfo(
          label: 'RECOMMEND',
          pathPoints: _createVariantRoute(dummyPathPoints, 1),
          displayPathPoints: dummyDisplayPoints,
          distanceM: 2050.0,
          durationSec: 1680, // 28분
          laneChanges: 4,
          uTurns: 1,
        ),
        'MAIN_ROAD': RouteInfo(
          label: 'MAIN_ROAD',
          pathPoints: _createVariantRoute(dummyPathPoints, 2),
          displayPathPoints: dummyDisplayPoints,
          distanceM: 1980.0,
          durationSec: 1500, // 25분
          laneChanges: 6,
          uTurns: 0,
        ),
      },
      requestEcho: {
        'origin': {'x': 129.0756, 'y': 35.1171},
        'destination': {'x': 129.0720, 'y': 35.1240},
      },
      elapsedMs: 250.0,
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
        case 1: // RECOMMEND
          offsetLng = offsetFactor;
          offsetLat = offsetFactor * 0.5;
          break;
        case 2: // MAIN_ROAD
          offsetLng = offsetFactor * 1.5;
          offsetLat = -offsetFactor * 0.5;
          break;
        default: // EASY
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

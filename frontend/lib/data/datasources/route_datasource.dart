import 'dart:convert';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/models/route_info_model.dart';
import 'package:http/http.dart' as http;
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_datasource.g.dart';

@riverpod
RouteDatasource routeDatasource(Ref ref) {
  return RouteDatasource();
}

class RouteDatasource {
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

  Future<List<RouteInfoModel>> fetchRoutes({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) async {
    try {
      final url = '$baseUrl/find-path';
      final response = await http
          .post(
            Uri.parse(url),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'origin': {'x': startLng, 'y': startLat},
              'destination': {'x': endLng, 'y': endLat},
            }),
          )
          .timeout(
            const Duration(seconds: 30),
            onTimeout: () {
              throw Exception('API 호출 타임아웃 (30초)');
            },
          );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        final routesData =
            responseData['routes'] as Map<String, dynamic>? ?? {};
        return routesData.entries
            .map(
              (entry) =>
                  RouteInfoModel.fromJson(entry.value as Map<String, dynamic>),
            )
            .toList();
      } else {
        throw Exception(
          'Route API Error: ${response.statusCode} - ${response.body}',
        );
      }
    } catch (e) {
      rethrow;
    }
  }
}

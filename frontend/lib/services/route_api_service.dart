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
        return RouteResponse.fromJson(responseData);
      } else {
        throw Exception('Failed to get route: ${response.statusCode}');
      }
    } catch (e) {
      print('API 에러: $e');
      throw Exception('Network error: $e');
    }
  }
}

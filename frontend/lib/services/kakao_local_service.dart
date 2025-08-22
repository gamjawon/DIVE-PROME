import 'dart:convert';

import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:frontend/models/place_model.dart';
import 'package:http/http.dart' as http;

class KakaoLocalService {
  static const String _baseUrl = 'https://dapi.kakao.com/v2/local';
  static String get _apiKey => dotenv.env['KAKAO_REST_API_KEY'] ?? '';

  // 키워드로 장소 검색
  Future<PlaceSearchResponse> searchPlaces({
    required String query,
    int page = 1,
    int size = 15,
  }) async {
    try {
      final uri = Uri.parse('$_baseUrl/search/keyword.json').replace(
        queryParameters: {
          'query': query,
          'page': page.toString(),
          'size': size.toString(),
        },
      );

      final response = await http.get(
        uri,
        headers: {
          'Authorization': 'KakaoAK $_apiKey',
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        return PlaceSearchResponse.fromJson(data);
      } else {
        throw Exception('Failed to search places: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error searching places: $e');
    }
  }

  // 좌표로 주소 검색
  Future<String> getAddressFromCoordinates({
    required double longitude,
    required double latitude,
  }) async {
    try {
      final uri = Uri.parse('$_baseUrl/geo/coord2address.json').replace(
        queryParameters: {'x': longitude.toString(), 'y': latitude.toString()},
      );

      final response = await http.get(
        uri,
        headers: {
          'Authorization': 'KakaoAK $_apiKey',
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        final documents = data['documents'] as List<dynamic>;

        if (documents.isNotEmpty) {
          final address = documents[0]['address'];
          return address['address_name'] ?? '';
        }
        return '';
      } else {
        throw Exception('Failed to get address: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error getting address: $e');
    }
  }
}

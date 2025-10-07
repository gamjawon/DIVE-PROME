import 'dart:convert';

import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/models/place_search_model.dart';
import 'package:http/http.dart' as http;
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'kakao_local_datasource.g.dart';

@riverpod
KakaoLocalDatasource kakaoLocalDatasource(Ref ref) {
  return KakaoLocalDatasource();
}

class KakaoLocalDatasource {
  static const String _baseUrl = 'https://dapi.kakao.com/v2/local';
  static final String _apiKey = dotenv.env['KAKAO_REST_API_KEY'] ?? '';

  Future<String?> fetchAddressFromCoordinates({
    required double longitude,
    required double latitude,
  }) async {
    try {
      final uri = Uri.parse('$_baseUrl/geo/coord2address.json').replace(
        queryParameters: {'x': longitude.toString(), 'y': latitude.toString()},
      );

      final response = await http.get(
        uri,
        headers: {'Authorization': 'KakaoAK $_apiKey'},
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        final documents = data['documents'] as List<dynamic>;

        if (documents.isNotEmpty) {
          final address = documents[0]['address'];
          return address['address_name'];
        }
        return null;
      } else {
        throw Exception(
          'Kakao API Error: ${response.statusCode} - ${response.body}',
        );
      }
    } catch (e) {
      rethrow;
    }
  }

  Future<PlaceSearchModel> fetchPlacesFromQuery({
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
        headers: {'Authorization': 'KakaoAK $_apiKey'},
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        return PlaceSearchModel.fromJson(data);
      } else {
        throw Exception(
          'Kakao API Error: ${response.statusCode} - ${response.body}',
        );
      }
    } catch (e) {
      rethrow;
    }
  }
}

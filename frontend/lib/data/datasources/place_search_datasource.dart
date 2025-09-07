import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/kakao_local_datasource.dart';
import 'package:frontend/data/models/place_search_model.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_datasource.g.dart';

@Riverpod(keepAlive: true)
PlaceSearchDatasource placeSearchDatasource(Ref ref) {
  return PlaceSearchDatasource(ref.watch(kakaoLocalDatasourceProvider));
}

class PlaceSearchDatasource {
  final KakaoLocalDatasource kakaoLocalDatasource;

  PlaceSearchDatasource(this.kakaoLocalDatasource);

  Future<PlaceSearchResponse> searchPlaces(String query) async {
    try {
      return await kakaoLocalDatasource.searchPlacesFromQuery(query: query);
    } catch (e) {
      print('API 오류로 빈 결과 반환: $e');
      // API 오류시 빈 결과 반환
      return const PlaceSearchResponse(documents: []);
    }
  }
}

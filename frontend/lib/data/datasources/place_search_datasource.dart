import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/kakao_local_datasource.dart';
import 'package:frontend/data/models/place_search_model.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_datasource.g.dart';

@riverpod
PlaceSearchDatasource placeSearchDatasource(Ref ref) {
  return PlaceSearchDatasource(ref.watch(kakaoLocalDatasourceProvider));
}

class PlaceSearchDatasource {
  final KakaoLocalDatasource kakaoLocalDatasource;

  PlaceSearchDatasource(this.kakaoLocalDatasource);

  Future<PlaceSearchModel> fetchPlaces(String query) async {
    try {
      return await kakaoLocalDatasource.fetchPlacesFromQuery(query: query);
    } catch (e) {
      rethrow;
    }
  }
}

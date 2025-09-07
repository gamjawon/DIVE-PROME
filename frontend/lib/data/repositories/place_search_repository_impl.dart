import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/place_search_datasource.dart';
import 'package:frontend/data/models/place_search_model.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_repository_impl.g.dart';

@Riverpod(keepAlive: true)
PlaceSearchRepositoryImpl placeSearchRepository(Ref ref) {
  return PlaceSearchRepositoryImpl(ref.watch(placeSearchDatasourceProvider));
}

class PlaceSearchRepositoryImpl {
  final PlaceSearchDatasource placeSearchDatasource;

  PlaceSearchRepositoryImpl(this.placeSearchDatasource);

  /// 키워드로 장소 검색하기
  Future<PlaceSearchResponse> searchPlaces(String query) async {
    return placeSearchDatasource.searchPlaces(query);
  }
}

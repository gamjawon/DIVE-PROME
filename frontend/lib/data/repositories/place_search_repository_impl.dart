import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/place_search_datasource.dart';
import 'package:frontend/data/models/place_search_model.dart';
import 'package:frontend/domain/entities/location.dart';
import 'package:frontend/domain/repositories/place_search_repository.dart';
import 'package:frontend/domain/value_objects/search_result.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_repository_impl.g.dart';

@riverpod
PlaceSearchRepository placeSearchRepository(Ref ref) {
  return PlaceSearchRepositoryImpl(ref.watch(placeSearchDatasourceProvider));
}

class PlaceSearchRepositoryImpl implements PlaceSearchRepository {
  final PlaceSearchDatasource placeSearchDatasource;

  PlaceSearchRepositoryImpl(this.placeSearchDatasource);

  @override
  Future<SearchResult<Location>> getPlaces(String query) async {
    try {
      final places = await placeSearchDatasource.fetchPlaces(query);
      return places.toEntity();
    } catch (e) {
      print('장소 검색하는 중 에러 발생: $e');
      rethrow;
    }
  }
}

import 'package:frontend/data/repositories/place_search_repository_impl.dart';
import 'package:frontend/domain/entities/location.dart';
import 'package:frontend/domain/value_objects/search_result.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_viewmodel.g.dart';

@riverpod
class PlaceSearchViewmodel extends _$PlaceSearchViewmodel {
  @override
  FutureOr<SearchResult<Location>> build() async {
    return _searchPlaces('');
  }

  Future<void> searchPlaces(String query) async {
    state = const AsyncValue.loading();
    state = await AsyncValue.guard(() async {
      return _searchPlaces(query);
    });
  }

  Future<SearchResult<Location>> _searchPlaces(String query) async {
    return await ref.read(placeSearchRepositoryProvider).getPlaces(query);
  }
}

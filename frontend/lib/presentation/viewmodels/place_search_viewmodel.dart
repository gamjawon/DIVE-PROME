import 'package:frontend/data/models/place_search_model.dart';
import 'package:frontend/data/repositories/place_search_repository_impl.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_viewmodel.g.dart';

@riverpod
class PlaceSearchViewmodel extends _$PlaceSearchViewmodel {
  @override
  AsyncValue<PlaceSearchResponse?> build() {
    return const AsyncValue.data(null);
  }

  Future<void> searchPlaces(String query) async {
    if (query.trim().isEmpty) {
      state = const AsyncValue.data(null);
      return;
    }

    state = const AsyncValue.loading();

    state = await AsyncValue.guard(() async {
      final repository = ref.read(placeSearchRepositoryProvider);
      final response = await repository.searchPlaces(query);
      return response;
    });
  }
}

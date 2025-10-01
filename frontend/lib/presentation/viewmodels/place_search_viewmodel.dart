import 'package:frontend/data/models/place_search_model.dart';
import 'package:frontend/data/repositories/place_search_repository_impl.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_viewmodel.g.dart';

// ==================== 장소 검색 결과 관리 ====================
@riverpod
class PlaceSearchViewmodel extends _$PlaceSearchViewmodel {
  @override
  AsyncValue<PlaceSearchResponse?> build() {
    return const AsyncValue.data(null);
  }

  /// 장소 검색
  Future<void> searchPlaces(String query) async {
    if (query.trim().isEmpty) {
      state = const AsyncValue.data(null);
      return;
    }

    state = const AsyncValue.loading();

    try {
      final repository = ref.read(placeSearchRepositoryProvider);
      final response = await repository.searchPlaces(query);
      state = AsyncValue.data(response);
    } catch (e, stackTrace) {
      state = AsyncValue.error(e, stackTrace);
    }
  }

  /// 검색 결과 초기화
  void clearSearchResults() {
    state = const AsyncValue.data(null);
  }
}

import 'package:frontend/data/models/location_model.dart';
import 'package:frontend/data/repositories/place_search_repository_impl.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_search_viewmodel.g.dart';

@Riverpod(keepAlive: true)
class PlaceSearchViewmodel extends _$PlaceSearchViewmodel {
  @override
  PlaceSearchState build() {
    return PlaceSearchState(
      selectedPlaces: {'start': null, 'end': null},
      searchResults: const AsyncValue.data([]),
    );
  }

  // ==================== 장소 선택 관련 ====================
  void setStartPlace(Location? place) {
    state = state.copyWith(
      selectedPlaces: {...state.selectedPlaces, 'start': place},
    );
  }

  void setEndPlace(Location? place) {
    state = state.copyWith(
      selectedPlaces: {...state.selectedPlaces, 'end': place},
    );
  }

  void swapPlaces() {
    final temp = state.selectedPlaces['start'];
    state = state.copyWith(
      selectedPlaces: {'start': state.selectedPlaces['end'], 'end': temp},
    );
  }

  void clearPlaces() {
    state = state.copyWith(selectedPlaces: {'start': null, 'end': null});
  }

  // ==================== 장소 검색 관련 ====================
  /// 장소 검색
  Future<void> searchPlaces(String query) async {
    if (query.trim().isEmpty) {
      state = state.copyWith(searchResults: const AsyncValue.data([]));
      return;
    }

    state = state.copyWith(searchResults: const AsyncValue.loading());

    try {
      final repository = ref.read(placeSearchRepositoryProvider);
      final response = await repository.searchPlaces(query);
      state = state.copyWith(
        searchResults: AsyncValue.data(response.documents),
      );
    } catch (e, stackTrace) {
      state = state.copyWith(searchResults: AsyncValue.error(e, stackTrace));
    }
  }

  /// 검색 결과 초기화
  void clearSearchResults() {
    state = state.copyWith(searchResults: const AsyncValue.data([]));
  }

  // ==================== Getters ====================
  Location? get startPlace => state.selectedPlaces['start'];
  Location? get endPlace => state.selectedPlaces['end'];
  AsyncValue<List<Location>> get searchResults => state.searchResults;
}

// PlaceSearch 상태를 나타내는 클래스
class PlaceSearchState {
  final Map<String, Location?> selectedPlaces;
  final AsyncValue<List<Location>> searchResults;

  const PlaceSearchState({
    required this.selectedPlaces,
    required this.searchResults,
  });

  PlaceSearchState copyWith({
    Map<String, Location?>? selectedPlaces,
    AsyncValue<List<Location>>? searchResults,
  }) {
    return PlaceSearchState(
      selectedPlaces: selectedPlaces ?? this.selectedPlaces,
      searchResults: searchResults ?? this.searchResults,
    );
  }

  // Getter methods for easier access
  Location? get startPlace => selectedPlaces['start'];
  Location? get endPlace => selectedPlaces['end'];
}

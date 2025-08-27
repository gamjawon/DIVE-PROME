import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/models/place_model.dart';

class PlaceNotifier extends StateNotifier<Map<String, SelectedPlace?>> {
  PlaceNotifier() : super({'start': null, 'end': null});

  void setStartPlace(SelectedPlace? place) {
    state = {...state, 'start': place};
  }

  void setEndPlace(SelectedPlace? place) {
    state = {...state, 'end': place};
  }

  void swapPlaces() {
    final temp = state['start'];
    state = {'start': state['end'], 'end': temp};
  }

  void clearPlaces() {
    state = {'start': null, 'end': null};
  }

  SelectedPlace? get startPlace => state['start'];
  SelectedPlace? get endPlace => state['end'];
}

final placeProvider =
    StateNotifierProvider<PlaceNotifier, Map<String, SelectedPlace?>>((ref) {
      return PlaceNotifier();
    });

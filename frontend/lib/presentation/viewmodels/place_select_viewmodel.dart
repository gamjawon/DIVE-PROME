import 'package:frontend/data/models/location_model.dart';
import 'package:frontend/presentation/states/selected_places.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'place_select_viewmodel.g.dart';

@Riverpod(keepAlive: true)
class PlaceSelectViewmodel extends _$PlaceSelectViewmodel {
  @override
  SelectedPlaces build() {
    return const SelectedPlaces();
  }

  void setStartPlace(Location? place) {
    state = state.copyWith(start: place);
  }

  void setEndPlace(Location? place) {
    state = state.copyWith(end: place);
  }

  void swapPlaces() {
    state = state.copyWith(start: state.end, end: state.start);
  }
}

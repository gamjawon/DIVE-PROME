import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/data/models/location_model.dart';

part 'selected_places.freezed.dart';

@freezed
abstract class SelectedPlaces with _$SelectedPlaces {
  const factory SelectedPlaces({Location? start, Location? end}) =
      _SelectedPlaces;
}

import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/data/models/location_model.dart';

part 'place_search_model.freezed.dart';
part 'place_search_model.g.dart';

@freezed
abstract class PlaceSearchResponse with _$PlaceSearchResponse {
  const factory PlaceSearchResponse({required List<Location> documents}) =
      _PlaceSearchResponse;

  factory PlaceSearchResponse.fromJson(Map<String, dynamic> json) =>
      _$PlaceSearchResponseFromJson(json);
}

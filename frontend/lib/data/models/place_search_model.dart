import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/data/models/location_model.dart';
import 'package:frontend/domain/entities/location.dart';
import 'package:frontend/domain/value_objects/search_result.dart';

part 'place_search_model.freezed.dart';
part 'place_search_model.g.dart';

@freezed
abstract class PlaceSearchModel with _$PlaceSearchModel {
  const factory PlaceSearchModel({
    @JsonKey(name: 'documents') required List<LocationModel> places,
  }) = _PlaceSearchResponse;

  factory PlaceSearchModel.fromJson(Map<String, dynamic> json) =>
      _$PlaceSearchResponseFromJson(json);
}

extension PlaceSearchModelX on PlaceSearchModel {
  SearchResult<Location> toEntity() =>
      SearchResult<Location>(items: places.map((e) => e.toEntity()).toList());
}

import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/data/utils/json_converters.dart';
import 'package:frontend/domain/entities/location.dart';

part 'location_model.freezed.dart';
part 'location_model.g.dart';

@freezed
abstract class LocationModel with _$LocationModel {
  const factory LocationModel({
    @JsonKey(name: 'y', fromJson: JsonConverters.toDouble)
    required double latitude,
    @JsonKey(name: 'x', fromJson: JsonConverters.toDouble)
    required double longitude,
    @JsonKey(name: 'place_name') String? placeName,
    @JsonKey(name: 'address_name') String? addressName,
  }) = _Location;

  factory LocationModel.fromJson(Map<String, dynamic> json) =>
      _$LocationFromJson(json);
}

extension LocationModelX on LocationModel {
  Location toEntity({
    String fallbackPlaceName = '알 수 없는 장소',
    String fallbackAddressName = '주소 없음',
  }) => Location(
    latitude: latitude,
    longitude: longitude,
    placeName: placeName ?? fallbackPlaceName,
    addressName: addressName ?? fallbackAddressName,
  );
}

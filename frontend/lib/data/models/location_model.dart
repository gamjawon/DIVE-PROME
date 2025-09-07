import 'package:freezed_annotation/freezed_annotation.dart';

part 'location_model.freezed.dart';
part 'location_model.g.dart';

@freezed
abstract class Location with _$Location {
  const factory Location({
    @JsonKey(name: 'y', fromJson: _stringToDouble) required double latitude,
    @JsonKey(name: 'x', fromJson: _stringToDouble) required double longitude,
    @JsonKey(name: 'place_name') required String placeName,
    @JsonKey(name: 'address_name') required String addressName,
    @JsonKey(name: 'road_address_name') @Default('') String roadAddressName,
    @JsonKey(name: 'category_name') @Default('') String categoryName,
  }) = _Location;

  factory Location.fromJson(Map<String, dynamic> json) =>
      _$LocationFromJson(json);
}

// 문자열 -> double 변환
double _stringToDouble(dynamic value) {
  if (value is double) return value;
  if (value is int) return value.toDouble();
  if (value is String) return double.parse(value);
  throw ArgumentError('Cannot convert $value to double');
}

extension LocationExtension on Location {
  // 검색 결과 표시용 주소 (도로명 주소 우선, 없으면 지번 주소)
  String get displayAddress =>
      roadAddressName.isNotEmpty ? roadAddressName : addressName;
}

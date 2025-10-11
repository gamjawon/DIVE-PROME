import 'package:freezed_annotation/freezed_annotation.dart';

part 'location.freezed.dart';

@freezed
abstract class Location with _$Location {
  const factory Location({
    required double latitude,
    required double longitude,
    required String placeName,
    required String addressName,
  }) = _Location;
}

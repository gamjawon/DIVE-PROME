// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'location_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_Location _$LocationFromJson(Map<String, dynamic> json) => _Location(
  latitude: _stringToDouble(json['y']),
  longitude: _stringToDouble(json['x']),
  placeName: json['place_name'] as String,
  addressName: json['address_name'] as String,
  roadAddressName: json['road_address_name'] as String? ?? '',
  categoryName: json['category_name'] as String? ?? '',
);

Map<String, dynamic> _$LocationToJson(_Location instance) => <String, dynamic>{
  'y': instance.latitude,
  'x': instance.longitude,
  'place_name': instance.placeName,
  'address_name': instance.addressName,
  'road_address_name': instance.roadAddressName,
  'category_name': instance.categoryName,
};

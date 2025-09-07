// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'place_search_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_PlaceSearchResponse _$PlaceSearchResponseFromJson(Map<String, dynamic> json) =>
    _PlaceSearchResponse(
      documents: (json['documents'] as List<dynamic>)
          .map((e) => Location.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$PlaceSearchResponseToJson(
  _PlaceSearchResponse instance,
) => <String, dynamic>{'documents': instance.documents};

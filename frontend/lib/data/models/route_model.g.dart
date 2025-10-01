// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'route_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_RouteInfo _$RouteInfoFromJson(Map<String, dynamic> json) => _RouteInfo(
  label: json['label'] as String,
  pathPoints:
      (json['path_points'] as List<dynamic>?)
          ?.map(
            (e) =>
                (e as List<dynamic>).map((e) => (e as num).toDouble()).toList(),
          )
          .toList() ??
      const [],
  displayPathPoints:
      (json['display_path_points'] as List<dynamic>?)
          ?.map(
            (e) =>
                (e as List<dynamic>).map((e) => (e as num).toDouble()).toList(),
          )
          .toList() ??
      const [],
  distanceM: (json['distance_m'] as num?)?.toDouble() ?? 0.0,
  durationSec: (json['duration_sec'] as num?)?.toInt() ?? 0,
  laneChanges: (json['lane_changes'] as num?)?.toInt() ?? 0,
  uTurns: (json['u_turns'] as num?)?.toInt() ?? 0,
  steepSlopes: (json['steep_slopes'] as num?)?.toInt() ?? 0,
);

Map<String, dynamic> _$RouteInfoToJson(_RouteInfo instance) =>
    <String, dynamic>{
      'label': instance.label,
      'path_points': instance.pathPoints,
      'display_path_points': instance.displayPathPoints,
      'distance_m': instance.distanceM,
      'duration_sec': instance.durationSec,
      'lane_changes': instance.laneChanges,
      'u_turns': instance.uTurns,
      'steep_slopes': instance.steepSlopes,
    };

_RouteRequest _$RouteRequestFromJson(Map<String, dynamic> json) =>
    _RouteRequest(
      startLat: (json['startLat'] as num).toDouble(),
      startLng: (json['startLng'] as num).toDouble(),
      endLat: (json['endLat'] as num).toDouble(),
      endLng: (json['endLng'] as num).toDouble(),
    );

Map<String, dynamic> _$RouteRequestToJson(_RouteRequest instance) =>
    <String, dynamic>{
      'startLat': instance.startLat,
      'startLng': instance.startLng,
      'endLat': instance.endLat,
      'endLng': instance.endLng,
    };

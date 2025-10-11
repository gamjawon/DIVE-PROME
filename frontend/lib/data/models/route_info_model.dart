import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/data/utils/json_converters.dart';
import 'package:frontend/domain/entities/route_info.dart';
import 'package:frontend/domain/enums/route_option.dart';

part 'route_info_model.freezed.dart';
part 'route_info_model.g.dart';

@freezed
abstract class RouteInfoModel with _$RouteInfoModel {
  const factory RouteInfoModel({
    @JsonKey(name: 'label') required String label,
    @JsonKey(name: 'path_points', fromJson: _convertPathPoints)
    @Default([])
    List<Map<String, double>> pathPoints,
    @JsonKey(name: 'distance_m') @Default(0.0) double distanceM,
    @JsonKey(name: 'duration_sec') @Default(0) int durationSec,
    @JsonKey(name: 'lane_changes') @Default(0) int laneChanges,
    @JsonKey(name: 'u_turns') @Default(0) int uTurns,
    @JsonKey(name: 'steep_slopes') @Default(0) int steepSlopes,
  }) = _RouteInfo;

  factory RouteInfoModel.fromJson(Map<String, dynamic> json) =>
      _$RouteInfoFromJson(json);
}

List<Map<String, double>> _convertPathPoints(List<dynamic> points) =>
    points.map((point) {
      return {
        'lng': JsonConverters.toDouble(point[0]),
        'lat': JsonConverters.toDouble(point[1]),
      };
    }).toList();

extension RouteInfoModelX on RouteInfoModel {
  RouteInfo toEntity() => RouteInfo(
    option: RouteOption.fromValue(label),
    pathPoints: pathPoints,
    distanceM: distanceM,
    durationSec: durationSec,
    laneChanges: laneChanges,
    uTurns: uTurns,
    steepSlopes: steepSlopes,
  );
}

import 'package:freezed_annotation/freezed_annotation.dart';

part 'route_model.freezed.dart';
part 'route_model.g.dart';

enum RouteOption {
  easy('EASY', '쉬운 길 추천'),
  recommend('RECOMMEND', '내비 추천'),
  mainRoad('MAIN_ROAD', '큰길 우선');

  const RouteOption(this.value, this.displayName);
  final String value;
  final String displayName;
}

@freezed
abstract class RouteInfo with _$RouteInfo {
  const factory RouteInfo({
    required String label,
    @JsonKey(name: 'path_points') @Default([]) List<List<double>> pathPoints,
    @JsonKey(name: 'display_path_points')
    @Default([])
    List<List<double>> displayPathPoints,
    @JsonKey(name: 'distance_m') @Default(0.0) double distanceM,
    @JsonKey(name: 'duration_sec') @Default(0) int durationSec,
    @JsonKey(name: 'lane_changes') @Default(0) int laneChanges,
    @JsonKey(name: 'u_turns') @Default(0) int uTurns,
    @JsonKey(name: 'steep_slopes') @Default(0) int steepSlopes,
  }) = _RouteInfo;

  const RouteInfo._();

  factory RouteInfo.fromJson(Map<String, dynamic> json) =>
      _$RouteInfoFromJson(json);

  // 거리를 km로 변환
  double get distanceKm => distanceM / 1000.0;

  // 시간을 분으로 변환
  int get durationMin => (durationSec / 60).round();

  // RouteOption enum으로 변환
  RouteOption get option {
    switch (label) {
      case 'EASY':
        return RouteOption.easy;
      case 'RECOMMEND':
        return RouteOption.recommend;
      case 'MAIN_ROAD':
        return RouteOption.mainRoad;
      default:
        return RouteOption.easy;
    }
  }
}

// 백엔드 응답을 List<RouteInfo>로 변환하는 헬퍼 함수
List<RouteInfo> parseRoutesFromResponse(Map<String, dynamic> json) {
  final routesData = json['routes'] as Map<String, dynamic>? ?? {};
  return routesData.entries
      .map((entry) => RouteInfo.fromJson(entry.value as Map<String, dynamic>))
      .toList();
}

@freezed
abstract class RouteRequest with _$RouteRequest {
  const factory RouteRequest({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) = _RouteRequest;

  const RouteRequest._();

  factory RouteRequest.fromJson(Map<String, dynamic> json) =>
      _$RouteRequestFromJson(json);

  Map<String, dynamic> toRequestBody() {
    return {
      'origin': {'x': startLng, 'y': startLat},
      'destination': {'x': endLng, 'y': endLat},
    };
  }
}

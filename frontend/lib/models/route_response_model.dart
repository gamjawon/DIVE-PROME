enum RouteOption {
  easy('EASY', '쉬운 길 추천'),
  navi('NAVI', '내비 추천'),
  wide('WIDE', '큰길 우선');

  const RouteOption(this.value, this.displayName);
  final String value;
  final String displayName;
}

class RouteInfo {
  final RouteOption option;
  final List<List<double>> pathPoints;
  final double distance; // km
  final int duration; // 분
  final int laneChanges; // 차선 변경 횟수
  final bool hasUturn; // 유턴 유무
  final int steepRoads; // 급경사로 수

  const RouteInfo({
    required this.option,
    required this.pathPoints,
    required this.distance,
    required this.duration,
    required this.laneChanges,
    required this.hasUturn,
    required this.steepRoads,
  });

  factory RouteInfo.fromJson(Map<String, dynamic> json) {
    // option 파싱
    final optionStr = json['option'] as String? ?? 'EASY';
    final option = RouteOption.values.firstWhere(
      (e) => e.value == optionStr,
      orElse: () => RouteOption.easy,
    );

    return RouteInfo(
      option: option,
      pathPoints:
          (json['path_points'] as List<dynamic>?)
              ?.map(
                (point) => (point as List<dynamic>)
                    .map((coord) => (coord as num).toDouble())
                    .toList(),
              )
              .toList()
              .cast<List<double>>() ??
          [],
      distance: (json['distance'] as num?)?.toDouble() ?? 0.0,
      duration: (json['duration'] as num?)?.toInt() ?? 0,
      laneChanges: (json['lane_changes'] as num?)?.toInt() ?? 0,
      hasUturn: json['has_uturn'] as bool? ?? false,
      steepRoads: (json['steep_roads'] as num?)?.toInt() ?? 0,
    );
  }

  @override
  String toString() {
    return 'RouteInfo(option: ${option.displayName}, distance: ${distance}km, duration: ${duration}분)';
  }
}

class RouteResponse {
  final List<RouteInfo> routes;

  const RouteResponse({required this.routes});

  factory RouteResponse.fromJson(Map<String, dynamic> json) {
    return RouteResponse(
      routes:
          (json['routes'] as List<dynamic>?)
              ?.map(
                (route) => RouteInfo.fromJson(route as Map<String, dynamic>),
              )
              .toList() ??
          [],
    );
  }
}

enum RouteOption {
  easy('EASY', '쉬운 길 추천'),
  recommend('RECOMMEND', '내비 추천'),
  mainRoad('MAIN_ROAD', '큰길 우선');

  const RouteOption(this.value, this.displayName);
  final String value;
  final String displayName;
}

class RouteInfo {
  final String label;
  final List<List<double>> pathPoints; // [lon, lat] 형식
  final List<List<double>> displayPathPoints; // 시각화용 간소화된 경로
  final double distanceM; // 미터
  final int durationSec; // 초
  final int laneChanges; // 차선 변경 횟수
  final int uTurns; // 유턴 횟수

  const RouteInfo({
    required this.label,
    required this.pathPoints,
    required this.displayPathPoints,
    required this.distanceM,
    required this.durationSec,
    required this.laneChanges,
    required this.uTurns,
  });

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

  factory RouteInfo.fromJson(Map<String, dynamic> json) {
    return RouteInfo(
      label: json['label'] as String? ?? 'EASY',
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
      displayPathPoints:
          (json['display_path_points'] as List<dynamic>?)
              ?.map(
                (point) => (point as List<dynamic>)
                    .map((coord) => (coord as num).toDouble())
                    .toList(),
              )
              .toList()
              .cast<List<double>>() ??
          [],
      distanceM: (json['distance_m'] as num?)?.toDouble() ?? 0.0,
      durationSec: (json['duration_sec'] as num?)?.toInt() ?? 0,
      laneChanges: (json['lane_changes'] as num?)?.toInt() ?? 0,
      uTurns: (json['u_turns'] as num?)?.toInt() ?? 0,
    );
  }

  @override
  String toString() {
    return 'RouteInfo(label: $label, distance: ${distanceKm.toStringAsFixed(1)}km, duration: $durationMin분, laneChanges: $laneChanges, uTurns: $uTurns)';
  }
}

class RouteResponse {
  final Map<String, RouteInfo> routes;
  final Map<String, dynamic> requestEcho;
  final double elapsedMs;

  const RouteResponse({
    required this.routes,
    required this.requestEcho,
    required this.elapsedMs,
  });

  // 편의 메서드들
  RouteInfo? get easyRoute => routes['EASY'];
  RouteInfo? get recommendRoute => routes['RECOMMEND'];
  RouteInfo? get mainRoadRoute => routes['MAIN_ROAD'];

  List<RouteInfo> get routeList => routes.values.toList();

  factory RouteResponse.fromJson(Map<String, dynamic> json) {
    final routesData = json['routes'] as Map<String, dynamic>? ?? {};
    final Map<String, RouteInfo> parsedRoutes = {};

    for (final entry in routesData.entries) {
      parsedRoutes[entry.key] = RouteInfo.fromJson(
        entry.value as Map<String, dynamic>,
      );
    }

    return RouteResponse(
      routes: parsedRoutes,
      requestEcho: json['request_echo'] as Map<String, dynamic>? ?? {},
      elapsedMs: (json['elapsed_ms'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

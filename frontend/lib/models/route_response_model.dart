class RouteResponse {
  final List<List<double>> pathPoints;

  const RouteResponse({required this.pathPoints});

  factory RouteResponse.fromJson(Map<String, dynamic> json) {
    return RouteResponse(
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
    );
  }

  @override
  String toString() {
    return 'RouteResponse(pathPoints: ${pathPoints.length} points)';
  }
}

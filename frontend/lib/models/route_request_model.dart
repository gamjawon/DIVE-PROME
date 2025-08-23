class RouteRequest {
  final double startLat;
  final double startLng;
  final double endLat;
  final double endLng;

  const RouteRequest({
    required this.startLat,
    required this.startLng,
    required this.endLat,
    required this.endLng,
  });

  Map<String, dynamic> toJson() {
    return {
      'origin': {'x': startLng, 'y': startLat},
      'destination': {'x': endLng, 'y': endLat},
      'priority': 'EASY',
    };
  }
}

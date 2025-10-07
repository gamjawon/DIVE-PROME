import 'package:frontend/domain/entities/route_info.dart';

abstract class RouteRepository {
  Future<List<RouteInfo>> getRoutes({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  });
}

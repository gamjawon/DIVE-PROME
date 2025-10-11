import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/route_datasource.dart';
import 'package:frontend/data/models/route_info_model.dart';
import 'package:frontend/domain/entities/route_info.dart';
import 'package:frontend/domain/enums/route_option.dart';
import 'package:frontend/domain/repositories/route_repository.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_repository_impl.g.dart';

@riverpod
RouteRepository routeRepository(Ref ref) {
  return RouteRepositoryImpl(ref.watch(routeDatasourceProvider));
}

class RouteRepositoryImpl implements RouteRepository {
  final RouteDatasource routeDatasource;

  RouteRepositoryImpl(this.routeDatasource);

  @override
  Future<List<RouteInfo>> getRoutes({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) async {
    try {
      final routes = await routeDatasource.fetchRoutes(
        startLat: startLat,
        startLng: startLng,
        endLat: endLat,
        endLng: endLng,
      );
      return routes.map((e) => e.toEntity()).toList();
    } catch (e) {
      print('경로 찾는 중 에러 발생: $e');
      print('더미 데이터 반환');
      // API 실패 시 더미 데이터 반환
      return _getDummyRoutes();
    }
  }

  // API 실패 시 사용할 더미 데이터
  List<RouteInfo> _getDummyRoutes() {
    // 부산 지역 더미 경로 데이터 (부산역 → 서면 → 해운대)
    final dummyPathPoints = [
      {'lng': 129.0756, 'lat': 35.1171}, // 부산역
      {'lng': 129.0750, 'lat': 35.1180},
      {'lng': 129.0745, 'lat': 35.1190},
      {'lng': 129.0740, 'lat': 35.1200},
      {'lng': 129.0735, 'lat': 35.1210},
      {'lng': 129.0730, 'lat': 35.1220},
      {'lng': 129.0725, 'lat': 35.1230},
      {'lng': 129.0720, 'lat': 35.1240}, // 서면교차로 인근
      {'lng': 129.0715, 'lat': 35.1250},
      {'lng': 129.0710, 'lat': 35.1260},
      {'lng': 129.0800, 'lat': 35.1300},
      {'lng': 129.0900, 'lat': 35.1350},
      {'lng': 129.1050, 'lat': 35.1400},
      {'lng': 129.1150, 'lat': 35.1450},
      {'lng': 129.1250, 'lat': 35.1500},
      {'lng': 129.1350, 'lat': 35.1550},
      {'lng': 129.1450, 'lat': 35.1580},
      {'lng': 129.1550, 'lat': 35.1600}, // 해운대 인근
    ];

    return [
      RouteInfo(
        option: RouteOption.easy,
        pathPoints: dummyPathPoints,
        distanceM: 2200.0,
        durationSec: 1800, // 30분
        laneChanges: 3,
        uTurns: 0,
        steepSlopes: 0,
      ),
      RouteInfo(
        option: RouteOption.recommend,
        pathPoints: _createVariantRoute(dummyPathPoints, 1),
        distanceM: 2050.0,
        durationSec: 1680, // 28분
        laneChanges: 4,
        uTurns: 1,
        steepSlopes: 1,
      ),
      RouteInfo(
        option: RouteOption.mainRoad,
        pathPoints: _createVariantRoute(dummyPathPoints, 2),
        distanceM: 1980.0,
        durationSec: 1500, // 25분
        laneChanges: 6,
        uTurns: 0,
        steepSlopes: 1,
      ),
    ];
  }

  List<Map<String, double>> _createVariantRoute(
    List<Map<String, double>> baseRoute,
    int variant,
  ) {
    if (baseRoute.isEmpty) return baseRoute;

    return baseRoute.asMap().entries.map((entry) {
      final index = entry.key;
      final point = entry.value;

      // 시작점과 끝점은 변경하지 않음
      if (index == 0 || index == baseRoute.length - 1) {
        return point;
      }

      // 각 경로별로 약간씩 다른 좌표를 생성
      double offsetLng = 0.0;
      double offsetLat = 0.0;

      // 경로 중간 지점들에만 오프셋 적용
      final offsetFactor = (index / baseRoute.length) * 0.001;

      switch (variant) {
        case 1: // RECOMMEND
          offsetLng = offsetFactor;
          offsetLat = offsetFactor * 0.5;
          break;
        case 2: // MAIN_ROAD
          offsetLng = offsetFactor * 1.5;
          offsetLat = -offsetFactor * 0.5;
          break;
        default: // EASY
          offsetLng = 0.0;
          offsetLat = 0.0;
          break;
      }

      return {
        'lng': (point['lng'] ?? 0.0) + offsetLng, // 경도
        'lat': (point['lat'] ?? 0.0) + offsetLat, // 위도
      };
    }).toList();
  }
}

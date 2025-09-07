import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/route_datasource.dart';
import 'package:frontend/data/models/route_model.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_repository_impl.g.dart';

@Riverpod(keepAlive: true)
RouteRepositoryImpl routeRepository(Ref ref) {
  return RouteRepositoryImpl(ref.watch(routeDatasourceProvider));
}

class RouteRepositoryImpl {
  final RouteDatasource routeDatasource;

  RouteRepositoryImpl(this.routeDatasource);

  /// 경로 찾기
  Future<List<RouteInfo>> getRoute(RouteRequest request) async {
    return routeDatasource.getRoute(request);
  }
}

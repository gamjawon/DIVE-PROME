import 'package:frontend/data/models/route_model.dart';
import 'package:frontend/data/repositories/route_repository_impl.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_viewmodel.g.dart';

@riverpod
class RouteViewmodel extends _$RouteViewmodel {
  @override
  Future<List<RouteInfo>?> build() async {
    // 초기 상태는 null (경로 데이터 없음)
    return null;
  }

  /// 경로 검색
  Future<void> searchRoute({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) async {
    state = const AsyncValue.loading();

    state = await AsyncValue.guard(() async {
      final request = RouteRequest(
        startLat: startLat,
        startLng: startLng,
        endLat: endLat,
        endLng: endLng,
      );

      return ref.read(routeRepositoryProvider).getRoute(request);
    });
  }

  /// 경로 데이터 초기화
  void clearRoute() {
    state = const AsyncValue.data(null);
  }

  /// 현재 경로 새로고침
  Future<void> refresh() async {
    state = await AsyncValue.guard(() async {
      return state.value; // 현재 상태 유지하면서 새로고침
    });
  }
}

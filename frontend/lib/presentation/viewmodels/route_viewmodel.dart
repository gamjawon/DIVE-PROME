import 'package:frontend/data/models/route_model.dart';
import 'package:frontend/data/repositories/route_repository_impl.dart';
import 'package:frontend/presentation/states/route_state.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_viewmodel.g.dart';

@riverpod
class RouteViewmodel extends _$RouteViewmodel {
  @override
  Future<RouteState> build() async {
    return RouteState(selectedOption: RouteOption.easy, routeList: null);
  }

  /// 경로 검색: repository에서 받아와서 RouteState의 routeList로 설정
  Future<void> searchRoute({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) async {
    // 로딩 상태 표기
    state = const AsyncValue.loading();

    state = await AsyncValue.guard(() async {
      final request = RouteRequest(
        startLat: startLat,
        startLng: startLng,
        endLat: endLat,
        endLng: endLng,
      );

      final routes = await ref.read(routeRepositoryProvider).getRoute(request);

      // 현재 RouteState(있다면)를 유지하면서 routeList만 바꿔줌
      final current =
          state.value ??
          RouteState(selectedOption: RouteOption.easy, routeList: null);

      return current.copyWith(routeList: routes);
    });
  }

  /// 선택된 경로 옵션 변경
  void setSelectedOption(RouteOption option) {
    final current =
        state.value ??
        RouteState(selectedOption: RouteOption.easy, routeList: null);
    state = AsyncValue.data(current.copyWith(selectedOption: option));
  }

  /// 경로 데이터 초기화 (routeList -> null)
  void clearRoute() {
    final current =
        state.value ??
        RouteState(selectedOption: RouteOption.easy, routeList: null);
    state = AsyncValue.data(current.copyWith(routeList: null));
  }
}

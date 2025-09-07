import 'package:frontend/data/models/route_model.dart';
import 'package:frontend/data/repositories/route_repository_impl.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_state_viewmodel.g.dart';

// RouteScreen의 UI 상태를 관리하는 ViewModel
@riverpod
class RouteStateViewmodel extends _$RouteStateViewmodel {
  @override
  RouteScreenState build() {
    return RouteScreenState(selectedOption: RouteOption.easy, routeList: null);
  }

  /// 선택된 경로 옵션 변경
  void setSelectedOption(RouteOption option) {
    state = state.copyWith(selectedOption: option);
  }

  /// 경로 리스트 설정
  void setRouteList(List<RouteInfo>? routes) {
    state = state.copyWith(routeList: routes);
  }

  /// 경로 검색
  Future<void> searchRoute({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) async {
    try {
      final request = RouteRequest(
        startLat: startLat,
        startLng: startLng,
        endLat: endLat,
        endLng: endLng,
      );

      final routes = await ref.read(routeRepositoryProvider).getRoute(request);
      setRouteList(routes);
    } catch (e) {
      print('경로 검색 중 오류 발생: $e');
      setRouteList(null);
    }
  }
}

// RouteScreen의 상태를 나타내는 클래스
class RouteScreenState {
  final RouteOption selectedOption;
  final List<RouteInfo>? routeList;

  const RouteScreenState({
    required this.selectedOption,
    required this.routeList,
  });

  RouteScreenState copyWith({
    RouteOption? selectedOption,
    List<RouteInfo>? routeList,
  }) {
    return RouteScreenState(
      selectedOption: selectedOption ?? this.selectedOption,
      routeList: routeList ?? this.routeList,
    );
  }
}

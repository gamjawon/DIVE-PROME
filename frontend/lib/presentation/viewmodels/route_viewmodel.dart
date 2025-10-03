import 'package:frontend/data/models/route_model.dart';
import 'package:frontend/data/repositories/route_repository_impl.dart';
import 'package:frontend/presentation/states/route_state.dart';
import 'package:frontend/presentation/viewmodels/place_select_viewmodel.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_viewmodel.g.dart';

@riverpod
class RouteViewmodel extends _$RouteViewmodel {
  @override
  Future<RouteState> build() async {
    return RouteState(selectedOption: RouteOption.easy, routeList: null);
  }

  Future<void> searchRoute() async {
    state = const AsyncValue.loading();

    final selectedPlaces = ref.read(placeSelectViewmodelProvider);

    state = await AsyncValue.guard(() async {
      final request = RouteRequest(
        startLat: selectedPlaces.start!.latitude,
        startLng: selectedPlaces.start!.longitude,
        endLat: selectedPlaces.end!.latitude,
        endLng: selectedPlaces.end!.longitude,
      );

      final routes = await ref.read(routeRepositoryProvider).getRoute(request);

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
}

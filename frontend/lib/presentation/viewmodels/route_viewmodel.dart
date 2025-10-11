import 'package:frontend/data/repositories/route_repository_impl.dart';
import 'package:frontend/domain/entities/location.dart';
import 'package:frontend/domain/enums/route_option.dart';
import 'package:frontend/presentation/states/route_state.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'route_viewmodel.g.dart';

@riverpod
class RouteViewmodel extends _$RouteViewmodel {
  @override
  FutureOr<RouteState> build() async {
    return RouteState(selectedOption: RouteOption.easy, routes: []);
  }

  void setStartPlace(Location place) {
    state = AsyncValue.data(state.value!.copyWith(start: place));
  }

  void setEndPlace(Location place) {
    state = AsyncValue.data(state.value!.copyWith(end: place));
  }

  void swapPlaces() {
    final current = state.value!;
    state = AsyncValue.data(
      current.copyWith(start: current.end, end: current.start),
    );
  }

  bool canSearchRoutes() {
    if (state.isLoading) {
      return false;
    }
    final current = state.value!;
    return current.start != null && current.end != null;
  }

  Future<void> searchRoute() async {
    final current = state.value!;
    if (!canSearchRoutes()) {
      return;
    }
    state = AsyncValue.loading();
    state = await AsyncValue.guard(() async {
      final routes = await ref
          .read(routeRepositoryProvider)
          .getRoutes(
            startLat: current.start!.latitude,
            startLng: current.start!.longitude,
            endLat: current.end!.latitude,
            endLng: current.end!.longitude,
          );
      return current.copyWith(routes: routes, selectedOption: RouteOption.easy);
    });
  }

  void setSelectedOption(RouteOption option) {
    final current = state.value!;
    state = AsyncValue.data(current.copyWith(selectedOption: option));
  }
}

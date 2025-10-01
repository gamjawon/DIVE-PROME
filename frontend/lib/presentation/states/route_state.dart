import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/data/models/route_model.dart';

part 'route_state.freezed.dart';

@freezed
abstract class RouteState with _$RouteState {
  const factory RouteState({
    required RouteOption selectedOption,
    required List<RouteInfo>? routeList,
  }) = _SelectedRoute;
}

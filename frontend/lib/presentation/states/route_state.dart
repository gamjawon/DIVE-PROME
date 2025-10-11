import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/domain/entities/location.dart';
import 'package:frontend/domain/entities/route_info.dart';
import 'package:frontend/domain/enums/route_option.dart';

part 'route_state.freezed.dart';

@freezed
abstract class RouteState with _$RouteState {
  const factory RouteState({
    Location? start,
    Location? end,
    required RouteOption selectedOption,
    required List<RouteInfo> routes,
  }) = _SelectedRoute;
}

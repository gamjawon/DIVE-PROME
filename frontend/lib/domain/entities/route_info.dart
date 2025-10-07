import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:frontend/domain/enums/route_option.dart';

part 'route_info.freezed.dart';

@freezed
abstract class RouteInfo with _$RouteInfo {
  const factory RouteInfo({
    required RouteOption option,
    required List<Map<String, double>> pathPoints,
    required double distanceM,
    required int durationSec,
    required int laneChanges,
    required int uTurns,
    required int steepSlopes,
  }) = _RouteInfo;

  const RouteInfo._();

  // 거리를 km로 변환
  double get distanceKm => distanceM / 1000.0;

  // 시간을 분으로 변환
  int get durationMin => (durationSec / 60).round();
}

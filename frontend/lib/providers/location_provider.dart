import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/models/location_model.dart';
import 'package:frontend/services/location_service.dart';

/// 위치 상태 Provider
final locationNotifierProvider =
    AsyncNotifierProvider<LocationNotifier, LocationModel?>(
      () => LocationNotifier(),
    );

class LocationNotifier extends AsyncNotifier<LocationModel?> {
  @override
  Future<LocationModel?> build() async {
    return await _fetchCurrentLocation();
  }

  /// 현재 위치 새로고침
  Future<void> refresh() async {
    state = const AsyncValue.loading();
    final location = await _fetchCurrentLocation();
    state = AsyncValue.data(location);
  }

  /// 현재 위치 가져오기
  Future<LocationModel?> _fetchCurrentLocation() async {
    try {
      final location = await LocationService.getCurrentLocation();
      if (location == null) throw Exception('위치 정보를 가져올 수 없습니다');
      return location;
    } catch (e, st) {
      state = AsyncValue.error(e, st);
      return null;
    }
  }
}

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/location_datasource.dart';
import 'package:frontend/data/models/location_model.dart';

/// 위치 상태 Provider
final locationNotifierProvider =
    AsyncNotifierProvider<LocationNotifier, Location?>(
      () => LocationNotifier(),
    );

class LocationNotifier extends AsyncNotifier<Location?> {
  @override
  Future<Location?> build() async {
    return await _fetchCurrentLocation();
  }

  /// 현재 위치 새로고침
  Future<void> refresh() async {
    state = const AsyncValue.loading();
    final location = await _fetchCurrentLocation();
    state = AsyncValue.data(location);
  }

  /// 현재 위치 가져오기
  Future<Location?> _fetchCurrentLocation() async {
    try {
      final location = await LocationDatasource.getCurrentLocation();
      if (location == null) throw Exception('위치 정보를 가져올 수 없습니다');
      return location;
    } catch (e, st) {
      state = AsyncValue.error(e, st);
      return null;
    }
  }
}

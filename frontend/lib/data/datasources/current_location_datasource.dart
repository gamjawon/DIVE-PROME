import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/kakao_local_datasource.dart';
import 'package:frontend/data/models/location_model.dart';
import 'package:geolocator/geolocator.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'current_location_datasource.g.dart';

@riverpod
CurrentLocationDatasource currentLocationDatasource(Ref ref) {
  return CurrentLocationDatasource(ref.watch(kakaoLocalDatasourceProvider));
}

class CurrentLocationDatasource {
  final KakaoLocalDatasource kakaoLocalDatasource;

  CurrentLocationDatasource(this.kakaoLocalDatasource);

  Future<LocationModel> fetchCurrentLocation() async {
    try {
      // 권한 확인
      final hasPermission = await _checkAndRequestPermission();
      if (!hasPermission) {
        throw Exception('위치 권한이 필요합니다');
      }

      // 현재 위치 가져오기
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      final address = await _fetchAddress(
        position.latitude,
        position.longitude,
      );

      return LocationModel(
        latitude: position.latitude,
        longitude: position.longitude,
        placeName: '현재 위치',
        addressName: address,
      );
    } catch (e) {
      rethrow;
    }
  }

  Future<bool> _checkAndRequestPermission() async {
    // 위치 서비스가 활성화되어 있는지 확인
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      return false;
    }

    // 권한 상태 확인
    LocationPermission permission = await Geolocator.checkPermission();

    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        return false;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      return false;
    }

    return true;
  }

  Future<String?> _fetchAddress(double latitude, double longitude) async {
    try {
      return await kakaoLocalDatasource.fetchAddressFromCoordinates(
        longitude: longitude,
        latitude: latitude,
      );
    } catch (e) {
      rethrow;
    }
  }
}

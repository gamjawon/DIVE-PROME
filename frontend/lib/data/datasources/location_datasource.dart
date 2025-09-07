import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/kakao_local_datasource.dart';
import 'package:frontend/data/models/location_model.dart';
import 'package:geolocator/geolocator.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'location_datasource.g.dart';

@Riverpod(keepAlive: true)
LocationDatasource locationDatasource(Ref ref) {
  return LocationDatasource(ref.watch(kakaoLocalDatasourceProvider));
}

class LocationDatasource {
  final KakaoLocalDatasource kakaoLocalDatasource;

  LocationDatasource(this.kakaoLocalDatasource);

  /// 현재 위치 가져오기
  Future<Location?> getCurrentLocation() async {
    try {
      // 권한 확인
      bool hasPermission = await checkAndRequestPermission();
      if (!hasPermission) {
        throw Exception('위치 권한이 필요합니다');
      }

      // 현재 위치 가져오기
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      String address = await getAddressFromLatLng(
        position.latitude,
        position.longitude,
      );

      return Location(
        latitude: position.latitude,
        longitude: position.longitude,
        placeName: '현재 위치',
        addressName: address,
      );
    } catch (e) {
      print('위치 정보를 가져오는 중 오류 발생: $e');
      return null;
    }
  }

  /// 위치 권한 확인하고 요청
  Future<bool> checkAndRequestPermission() async {
    // 위치 서비스가 활성화되어 있는지 확인
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
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

  /// 위도 경도에 해당하는 주소 가져오기
  Future<String> getAddressFromLatLng(double latitude, double longitude) async {
    try {
      return await kakaoLocalDatasource.getAddressFromCoordinates(
        longitude: longitude,
        latitude: latitude,
      );
    } catch (e) {
      print('주소를 가져오는 중 오류 발생: $e');
      return "주소 알 수 없음";
    }
  }
}

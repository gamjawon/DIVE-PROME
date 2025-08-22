import 'package:frontend/models/location_model.dart';
import 'package:frontend/services/kakao_local_service.dart';
import 'package:geolocator/geolocator.dart';

class LocationService {
  /// 위치 권한 확인하고 요청
  static Future<bool> checkAndRequestPermission() async {
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

  /// 현재 위치 가져오기
  static Future<LocationModel?> getCurrentLocation() async {
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

      return LocationModel(
        latitude: position.latitude,
        longitude: position.longitude,
        address: address,
      );
    } catch (e) {
      print('위치 정보를 가져오는 중 오류 발생: $e');
      return null;
    }
  }

  /// 위도 경도에 해당하는 주소 가져오기
  static Future<String> getAddressFromLatLng(
    double latitude,
    double longitude,
  ) async {
    try {
      final kakaoLocalService = KakaoLocalService();
      return await kakaoLocalService.getAddressFromCoordinates(
        longitude: longitude,
        latitude: latitude,
      );
    } catch (e) {
      print('주소를 가져오는 중 오류 발생: $e');
      return "주소 알 수 없음";
    }
  }
}

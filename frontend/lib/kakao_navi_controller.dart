import 'package:flutter/services.dart';

class KakaoNaviController {
  late MethodChannel _channel;

  KakaoNaviController(int id) {
    _channel = MethodChannel('kakao_navi_view_$id');
  }

  Future<String> authenticate({
    String appKey = "932cbb4d2258634817f0290271ad43fa",
    String clientVersion = "1.0.0",
    String userKey = "testUser",
  }) async {
    try {
      final result = await _channel.invokeMethod('authenticate', {
        'appKey': appKey,
        'clientVersion': clientVersion,
        'userKey': userKey,
      });
      return result;
    } catch (e) {
      throw Exception('인증 실패: $e');
    }
  }

  Future<String> startNavigation({
    required double startLat,
    required double startLng,
    required double goalLat,
    required double goalLng,
    String startName = "출발지",
    String goalName = "목적지",
  }) async {
    try {
      final result = await _channel.invokeMethod('startNavigation', {
        'startLat': startLat,
        'startLng': startLng,
        'goalLat': goalLat,
        'goalLng': goalLng,
        'startName': startName,
        'goalName': goalName,
      });
      return result;
    } catch (e) {
      throw Exception('내비게이션 시작 실패: $e');
    }
  }
}

import 'package:flutter/painting.dart';
import 'package:frontend/domain/enums/route_option.dart';

class Palette {
  static const Color primaryAccentColor = Color(0xFFFF5930);

  // 경로별 색상 정의
  static const Map<RouteOption, Color> routeColors = {
    RouteOption.easy: Color(0xFFFF792B), // 주황색
    RouteOption.recommend: Color(0xFF4285F4), // 파란색
    RouteOption.mainRoad: Color(0xFF34A853), // 초록색
  };
  static const Color inactiveRouteColor = Color(0xFFD1D5DB);
}

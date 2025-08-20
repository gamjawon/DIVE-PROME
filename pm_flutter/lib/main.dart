import 'package:flutter/material.dart';
import 'package:kakao_flutter_sdk/kakao_flutter_sdk.dart';
import 'screens/nav_page.dart';  // nav_page.dart 임포트

void main() {
  // 실제 카카오 네이티브 앱 키로 교체하세요!
  KakaoSdk.init(nativeAppKey: 'e9b8c35c76d62a0447680b44e9f32452');
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '카카오 네비게이션 예시',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: NavigationScreen(), // nav_page.dart의 NavigationScreen 사용
    );
  }
}
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/presentation/screens/home_screen.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(fileName: ".env");
  await KakaoMapSdk.instance.initialize(dotenv.get('KAKAO_NATIVE_APP_KEY'));
  runApp(const ProviderScope(child: EasyNaviApp()));
}

class EasyNaviApp extends StatelessWidget {
  const EasyNaviApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: 'Easy Navi', home: HomeScreen());
  }
}

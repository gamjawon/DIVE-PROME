import 'package:flutter/material.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: KakaoMap(
        option: const KakaoMapOption(
          position: LatLng(37.5665, 126.978),
          zoomLevel: 16,
          mapType: MapType.normal,
        ),
        onMapReady: (KakaoMapController controller) {},
      ),
    );
  }
}

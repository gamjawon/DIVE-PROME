import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/presentation/viewmodels/current_location_viewmodel.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

class KakaoMapBackground extends ConsumerStatefulWidget {
  const KakaoMapBackground({super.key});

  @override
  ConsumerState<ConsumerStatefulWidget> createState() =>
      _KakaoMapBackgroundState();
}

class _KakaoMapBackgroundState extends ConsumerState<KakaoMapBackground> {
  KakaoMapController? _mapController;

  @override
  Widget build(BuildContext context) {
    final currentLocationStateAsync = ref.watch(
      currentLocationViewmodelProvider,
    );
    // 위치 갱신되면 지도 이동
    currentLocationStateAsync.whenData((location) {
      if (_mapController != null) {
        _mapController!.moveCamera(
          CameraUpdate.newCenterPosition(
            LatLng(location.latitude, location.longitude),
          ),
        );
      }
    });

    return KakaoMap(
      option: KakaoMapOption(
        position: const LatLng(37.5665, 126.978),
        zoomLevel: 16,
        mapType: MapType.normal,
      ),
      onMapReady: (controller) {
        _mapController = controller;
      },
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/presentation/viewmodels/home_viewmodel.dart';
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
  void initState() {
    super.initState();
    // 화면 초기 렌더 후 위치 가져오기
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final notifier = ref.read(locationViewmodelProvider.notifier);
      if (ref.read(locationViewmodelProvider).value == null) {
        notifier.refresh();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final locationState = ref.watch(locationViewmodelProvider);
    // 위치 갱신되면 지도 이동
    locationState.whenData((location) {
      if (_mapController != null) {
        _mapController!.moveCamera(
          CameraUpdate.newCenterPosition(
            LatLng(location!.latitude, location.longitude),
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

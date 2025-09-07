import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/presentation/viewmodels/home_viewmodel.dart';
import 'package:frontend/presentation/views/home/widgets/bottom_navigation_panel.dart';
import 'package:frontend/presentation/views/home/widgets/route_form.dart';
import 'package:frontend/presentation/views/home/widgets/top_status_bar.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
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
    final screenWidth = MediaQuery.sizeOf(context).width;

    final locationState = ref.watch(locationViewmodelProvider);
    // 위치 갱신되면 지도 이동
    locationState.whenData((location) {
      if (_mapController != null && location != null) {
        _mapController!.moveCamera(
          CameraUpdate.newCenterPosition(
            LatLng(location.latitude, location.longitude),
          ),
        );
      }
    });

    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Stack(
        children: [
          KakaoMap(
            option: KakaoMapOption(
              position: const LatLng(37.5665, 126.978),
              zoomLevel: 16,
              mapType: MapType.normal,
            ),
            onMapReady: (controller) {
              _mapController = controller;
            },
          ),
          Container(width: screenWidth, height: 100, color: Colors.white),
          Positioned(
            top: 0,
            left: 0,
            child: SafeArea(child: TopStatusBar(screenWidth: screenWidth)),
          ),
          Positioned(
            top: 170,
            left: 15,
            right: 15,
            child: RouteForm(screenWidth: screenWidth),
          ),
          Positioned(
            bottom: 20,
            left: 65,
            right: 65,
            child: SafeArea(child: BottomNavigationPanel()),
          ),
        ],
      ),
    );
  }
}

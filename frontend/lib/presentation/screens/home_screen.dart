import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:frontend/providers/location_provider.dart';
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
      final notifier = ref.read(locationNotifierProvider.notifier);
      if (ref.read(locationNotifierProvider).value == null) {
        notifier.refresh();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.sizeOf(context).width;

    final locationState = ref.watch(locationNotifierProvider);
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

class TopStatusBar extends ConsumerWidget {
  const TopStatusBar({super.key, required this.screenWidth});

  final double screenWidth;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final locationState = ref.watch(locationNotifierProvider);

    return Container(
      width: screenWidth,
      height: 100,
      decoration: ShapeDecoration(
        color: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
            bottomLeft: Radius.circular(32),
            bottomRight: Radius.circular(32),
          ),
        ),
      ),
      padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            width: 36.76,
            height: 36.76,
            decoration: ShapeDecoration(
              color: Colors.white,
              shape: OvalBorder(
                side: BorderSide(
                  width: 1.50,
                  strokeAlign: BorderSide.strokeAlignOutside,
                  color: const Color(0xFFD7D7D7),
                ),
              ),
            ),
            child: ClipOval(
              child: SvgPicture.asset(
                'assets/icons/profile.svg',
                width: 36.76,
                height: 36.76,
                fit: BoxFit.cover,
              ),
            ),
          ),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                '옥순님, 안녕하세요!',
                style: TextStyle(
                  color: Colors.black,
                  fontSize: 16,
                  fontFamily: 'Pretendard',
                  fontWeight: FontWeight.w600,
                ),
              ),
              Row(
                children: [
                  SvgPicture.asset(
                    'assets/icons/pin.svg',
                    width: 20,
                    height: 20,
                  ),
                  SizedBox(width: 4),
                  Text(
                    locationState.when(
                      data: (location) =>
                          location == null ? '위치 정보 없음' : location.address,
                      error: (error, stackTrace) => 'Error: $error',
                      loading: () => 'Loading...',
                    ),
                    style: TextStyle(
                      color: Colors.black,
                      fontSize: 16,
                      fontFamily: 'Pretendard',
                      fontWeight: FontWeight.w600,
                      height: 1.06,
                    ),
                  ),
                ],
              ),
            ],
          ),
          SvgPicture.asset('assets/icons/settings.svg', width: 32, height: 32),
        ],
      ),
    );
  }
}

class RouteForm extends StatelessWidget {
  const RouteForm({super.key, required this.screenWidth});

  final double screenWidth;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 200,
      decoration: ShapeDecoration(
        color: Colors.white,
        shape: RoundedRectangleBorder(
          side: BorderSide(width: 1, color: const Color(0xFFF0F0F0)),
          borderRadius: BorderRadius.circular(8),
        ),
        shadows: [
          BoxShadow(
            color: Color(0x3FA6A6A6),
            blurRadius: 4,
            offset: Offset(1, 1),
            spreadRadius: 0,
          ),
          BoxShadow(
            color: Color(0x3FDEDEDE),
            blurRadius: 4,
            offset: Offset(-1, -1),
            spreadRadius: 0,
          ),
        ],
      ),
      padding: EdgeInsets.all(15),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                width: screenWidth - 100,
                height: 50,
                padding: const EdgeInsets.symmetric(
                  horizontal: 21,
                  vertical: 9,
                ),
                decoration: ShapeDecoration(
                  color: Colors.white,
                  shape: RoundedRectangleBorder(
                    side: BorderSide(
                      width: 0.90,
                      color: const Color(0xFFBEBEBE),
                    ),
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.start,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  spacing: 10,
                  children: [
                    Text(
                      '부산역',
                      style: TextStyle(
                        color: const Color(0xFF374151),
                        fontSize: 16,
                        fontFamily: 'Pretendard',
                        fontWeight: FontWeight.w500,
                        height: 1.50,
                        letterSpacing: 0.09,
                      ),
                    ),
                  ],
                ),
              ),
              SvgPicture.asset('assets/icons/swap.svg', width: 25, height: 25),
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                width: screenWidth - 100,
                height: 50,
                padding: const EdgeInsets.symmetric(
                  horizontal: 21,
                  vertical: 9,
                ),
                decoration: ShapeDecoration(
                  color: Colors.white,
                  shape: RoundedRectangleBorder(
                    side: BorderSide(
                      width: 0.90,
                      color: const Color(0xFFBEBEBE),
                    ),
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.start,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  spacing: 10,
                  children: [
                    Text(
                      '서면 교차로',
                      style: TextStyle(
                        color: const Color(0xFF374151),
                        fontSize: 16,
                        fontFamily: 'Pretendard',
                        fontWeight: FontWeight.w500,
                        height: 1.50,
                        letterSpacing: 0.09,
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(
                width: 25,
                height: 25,
                child: SvgPicture.asset(
                  'assets/icons/more.svg',
                  width: 25,
                  height: 25,
                ),
              ),
            ],
          ),
          Container(
            width: double.infinity,
            height: 56,
            decoration: ShapeDecoration(
              gradient: LinearGradient(
                begin: Alignment(1.00, 0.50),
                end: Alignment(0.00, 0.50),
                colors: [const Color(0xFFFF5A31), const Color(0xFFFF792C)],
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              spacing: 10,
              children: [
                Text(
                  '길찾기',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontFamily: 'Pretendard',
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class BottomNavigationPanel extends StatelessWidget {
  const BottomNavigationPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 80,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      decoration: ShapeDecoration(
        color: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(36.27),
        ),
        shadows: [
          BoxShadow(
            color: Color(0x3F868686),
            blurRadius: 8,
            offset: Offset(2, 2),
            spreadRadius: 0,
          ),
          BoxShadow(
            color: Color(0x3FC4C4C4),
            blurRadius: 5,
            offset: Offset(-1, -1),
            spreadRadius: 0,
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        crossAxisAlignment: CrossAxisAlignment.center,
        spacing: 10,
        children: [
          SvgPicture.asset('assets/icons/home.svg', width: 30, height: 30),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 15),
            decoration: ShapeDecoration(
              color: const Color(0xFFFF5930),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(55),
              ),
              shadows: [
                BoxShadow(
                  color: Color(0x3FE63100),
                  blurRadius: 14,
                  offset: Offset(0, 0),
                  spreadRadius: 2,
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              spacing: 12,
              children: [
                SvgPicture.asset(
                  'assets/icons/navi.svg',
                  width: 30,
                  height: 30,
                ),
                Text(
                  '네비게이션',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 15,
                    fontFamily: 'Pretendard',
                    fontWeight: FontWeight.w500,
                    height: 1,
                  ),
                ),
              ],
            ),
          ),
          SvgPicture.asset('assets/icons/mypage.svg', width: 30, height: 30),
        ],
      ),
    );
  }
}

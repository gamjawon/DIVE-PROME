import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:frontend/models/place_model.dart';
import 'package:frontend/models/selected_place.dart';
import 'package:frontend/presentation/screens/place_search_screen.dart';
import 'package:frontend/presentation/screens/route_view_screen.dart';
import 'package:frontend/providers/location_provider.dart';
import 'package:frontend/providers/route_provider.dart';
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

class RouteForm extends ConsumerStatefulWidget {
  const RouteForm({super.key, required this.screenWidth});

  final double screenWidth;

  @override
  ConsumerState<RouteForm> createState() => _RouteFormState();
}

class _RouteFormState extends ConsumerState<RouteForm> {
  final TextEditingController _startController = TextEditingController();
  final TextEditingController _endController = TextEditingController();
  SelectedPlace? _startPlace;
  SelectedPlace? _endPlace;
  bool _isButtonEnabled = false;

  @override
  void initState() {
    super.initState();
    // 텍스트 변경 감지
    _startController.addListener(_updateButtonState);
    _endController.addListener(_updateButtonState);
  }

  @override
  void dispose() {
    _startController.dispose();
    _endController.dispose();
    super.dispose();
  }

  void _updateButtonState() {
    setState(() {
      _isButtonEnabled = _startPlace != null && _endPlace != null;
    });
  }

  void _swapLocations() {
    final tempPlace = _startPlace;
    final tempText = _startController.text;

    setState(() {
      _startPlace = _endPlace;
      _endPlace = tempPlace;
      _startController.text = _endController.text;
      _endController.text = tempText;
    });
    _updateButtonState();
  }

  Future<void> _selectStartPlace() async {
    final result = await Navigator.push<Place>(
      context,
      MaterialPageRoute(
        builder: (context) =>
            PlaceSearchScreen(title: '출발지 선택', hintText: '출발지를 검색하세요'),
      ),
    );

    if (result != null) {
      setState(() {
        _startPlace = SelectedPlace.fromPlace(result);
        _startController.text = _startPlace!.name;
      });
      _updateButtonState();
    }
  }

  Future<void> _findRoute() async {
    if (_startPlace == null || _endPlace == null) return;

    try {
      await ref
          .read(routeProvider.notifier)
          .getRoute(
            startLat: _startPlace!.latitude,
            startLng: _startPlace!.longitude,
            endLat: _endPlace!.latitude,
            endLng: _endPlace!.longitude,
          );

      // 경로 결과 출력
      final routeState = ref.read(routeProvider);
      routeState.whenData((response) {
        if (response != null && response.routes.isNotEmpty) {
          print('총 경로 개수: ${response.routes.length}');
          for (var route in response.routes) {
            print(
              '${route.option.displayName}: ${route.pathPoints.length}개 좌표, ${route.distance}km, ${route.duration}분',
            );
          }
        }
      });

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const RouteViewScreen()),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('경로를 찾을 수 없습니다: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _selectEndPlace() async {
    final result = await Navigator.push<Place>(
      context,
      MaterialPageRoute(
        builder: (context) =>
            PlaceSearchScreen(title: '도착지 선택', hintText: '도착지를 검색하세요'),
      ),
    );

    if (result != null) {
      setState(() {
        _endPlace = SelectedPlace.fromPlace(result);
        _endController.text = _endPlace!.name;
      });
      _updateButtonState();
    }
  }

  @override
  Widget build(BuildContext context) {
    final routeState = ref.watch(routeProvider);
    final isLoading = routeState.isLoading;

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
                width: widget.screenWidth - 100,
                height: 50,
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
                child: GestureDetector(
                  onTap: _selectStartPlace,
                  child: AbsorbPointer(
                    child: TextField(
                      controller: _startController,
                      decoration: InputDecoration(
                        hintText: '출발지를 입력하세요',
                        hintStyle: TextStyle(
                          color: const Color(0xFF9CA3AF),
                          fontSize: 16,
                          fontFamily: 'Pretendard',
                          fontWeight: FontWeight.w500,
                        ),
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 21,
                          vertical: 9,
                        ),
                      ),
                      style: TextStyle(
                        color: const Color(0xFF374151),
                        fontSize: 16,
                        fontFamily: 'Pretendard',
                        fontWeight: FontWeight.w500,
                        height: 1.50,
                        letterSpacing: 0.09,
                      ),
                    ),
                  ),
                ),
              ),
              GestureDetector(
                onTap: _swapLocations,
                child: SvgPicture.asset(
                  'assets/icons/swap.svg',
                  width: 25,
                  height: 25,
                ),
              ),
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                width: widget.screenWidth - 100,
                height: 50,
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
                child: GestureDetector(
                  onTap: _selectEndPlace,
                  child: AbsorbPointer(
                    child: TextField(
                      controller: _endController,
                      decoration: InputDecoration(
                        hintText: '도착지를 입력하세요',
                        hintStyle: TextStyle(
                          color: const Color(0xFF9CA3AF),
                          fontSize: 16,
                          fontFamily: 'Pretendard',
                          fontWeight: FontWeight.w500,
                        ),
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 21,
                          vertical: 9,
                        ),
                      ),
                      style: TextStyle(
                        color: const Color(0xFF374151),
                        fontSize: 16,
                        fontFamily: 'Pretendard',
                        fontWeight: FontWeight.w500,
                        height: 1.50,
                        letterSpacing: 0.09,
                      ),
                    ),
                  ),
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
              gradient: (_isButtonEnabled && !isLoading)
                  ? LinearGradient(
                      begin: Alignment(1.00, 0.50),
                      end: Alignment(0.00, 0.50),
                      colors: [
                        const Color(0xFFFF5A31),
                        const Color(0xFFFF792C),
                      ],
                    )
                  : null,
              color: (_isButtonEnabled && !isLoading)
                  ? null
                  : const Color(0xFFE5E7EB),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(8),
                onTap: (_isButtonEnabled && !isLoading) ? _findRoute : null,
                child: Center(
                  child: isLoading
                      ? SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.white,
                            ),
                          ),
                        )
                      : Text(
                          '길찾기',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: (_isButtonEnabled && !isLoading)
                                ? Colors.white
                                : const Color(0xFF9CA3AF),
                            fontSize: 18,
                            fontFamily: 'Pretendard',
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                ),
              ),
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
          GestureDetector(
            onTap: () {},
            child: Container(
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
          ),
          SvgPicture.asset('assets/icons/mypage.svg', width: 30, height: 30),
        ],
      ),
    );
  }
}

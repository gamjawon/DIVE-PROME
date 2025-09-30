import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:frontend/data/models/location_model.dart';
import 'package:frontend/presentation/viewmodels/place_search_viewmodel.dart';
import 'package:frontend/presentation/viewmodels/route_viewmodel.dart';
import 'package:frontend/presentation/views/route/route_screen.dart';
import 'package:frontend/presentation/views/search/place_search_screen.dart';

class RouteForm extends ConsumerStatefulWidget {
  const RouteForm({super.key, required this.screenWidth});

  final double screenWidth;

  @override
  ConsumerState<RouteForm> createState() => _RouteFormState();
}

class _RouteFormState extends ConsumerState<RouteForm> {
  final TextEditingController _startController = TextEditingController();
  final TextEditingController _endController = TextEditingController();
  Location? _startPlace;
  Location? _endPlace;
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

    // Place provider에도 업데이트
    ref.read(placeSearchViewmodelProvider.notifier).setStartPlace(_startPlace);
    ref.read(placeSearchViewmodelProvider.notifier).setEndPlace(_endPlace);

    _updateButtonState();
  }

  Future<void> _selectStartPlace() async {
    final result = await Navigator.push<Location>(
      context,
      MaterialPageRoute(
        builder: (context) =>
            PlaceSearchScreen(title: '출발지 선택', hintText: '출발지를 검색하세요'),
      ),
    );

    if (result != null) {
      setState(() {
        _startPlace = result;
        _startController.text = _startPlace!.placeName;
      });
      // Place provider에도 업데이트
      ref
          .read(placeSearchViewmodelProvider.notifier)
          .setStartPlace(_startPlace);
      _updateButtonState();
    }
  }

  Future<void> _selectEndPlace() async {
    final result = await Navigator.push<Location>(
      context,
      MaterialPageRoute(
        builder: (context) =>
            PlaceSearchScreen(title: '도착지 선택', hintText: '도착지를 검색하세요'),
      ),
    );

    if (result != null) {
      setState(() {
        _endPlace = result;
        _endController.text = _endPlace!.placeName;
      });
      // Place provider에도 업데이트
      ref.read(placeSearchViewmodelProvider.notifier).setEndPlace(_endPlace);
      _updateButtonState();
    }
  }

  Future<void> _findRoute() async {
    if (_startPlace == null || _endPlace == null) return;

    try {
      await ref
          .read(routeViewmodelProvider.notifier)
          .searchRoute(
            startLat: _startPlace!.latitude,
            startLng: _startPlace!.longitude,
            endLat: _endPlace!.latitude,
            endLng: _endPlace!.longitude,
          );

      // 경로 결과 출력
      final routeState = ref.read(routeViewmodelProvider);
      routeState.whenData((routeList) {
        if (routeList != null && routeList.isNotEmpty) {
          print('총 경로 개수: ${routeList.length}');
          for (var route in routeList) {
            print(
              '${route.option.displayName}: ${route.pathPoints.length}개 좌표, ${route.distanceKm}km, ${route.durationMin}분',
            );
          }
        }
      });

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const RouteScreen()),
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

  @override
  Widget build(BuildContext context) {
    final routeState = ref.watch(routeViewmodelProvider);
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

import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:kakao_map_sdk/kakao_map_sdk.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.sizeOf(context).width;

    return Scaffold(
      body: Stack(
        children: [
          KakaoMap(
            option: const KakaoMapOption(
              position: LatLng(37.5665, 126.978),
              zoomLevel: 16,
              mapType: MapType.normal,
            ),
            onMapReady: (KakaoMapController controller) {},
          ),

          Container(width: screenWidth, height: 100, color: Colors.white),
          Positioned(
            top: 0,
            left: 0,
            child: SafeArea(
              child: Container(
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
                              '부산, 서면',
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
                    SvgPicture.asset(
                      'assets/icons/settings.svg',
                      width: 32,
                      height: 32,
                    ),
                  ],
                ),
              ),
            ),
          ),
          Positioned(
            top: 170,
            left: 15,
            right: 15,
            child: Container(
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
                      SvgPicture.asset(
                        'assets/icons/swap.svg',
                        width: 25,
                        height: 25,
                      ),
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
                        colors: [
                          const Color(0xFFFF5A31),
                          const Color(0xFFFF792C),
                        ],
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
            ),
          ),
          Positioned(
            bottom: 20,
            left: 65,
            right: 65,
            child: SafeArea(
              child: Container(
                height: 80,
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 10,
                ),
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
                    SvgPicture.asset(
                      'assets/icons/home.svg',
                      width: 30,
                      height: 30,
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 22,
                        vertical: 15,
                      ),
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
                    SvgPicture.asset(
                      'assets/icons/mypage.svg',
                      width: 30,
                      height: 30,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

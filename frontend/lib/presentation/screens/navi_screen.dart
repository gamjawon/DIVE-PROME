import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class NaviScreen extends StatefulWidget {
  const NaviScreen({super.key});

  @override
  State<NaviScreen> createState() => _NaviScreenState();
}

class _NaviScreenState extends State<NaviScreen> {
  bool _isEasyMode = false;

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final easyBackgroundHeight = 320.0;
    final iconSize = _isEasyMode ? 160.0 : 80.0;
    final slideOffset = _isEasyMode ? easyBackgroundHeight : 0.0;

    return Scaffold(
      body: Stack(
        children: [
          // 배경 이미지
          Positioned.fill(
            child: Image.asset(
              "assets/images/navi_sample.png",
              fit: BoxFit.cover,
            ),
          ),
          // Easy 모드 배경 (GIF)
          AnimatedPositioned(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            top: 0,
            left: 0,
            width: screenSize.width,
            height: _isEasyMode ? easyBackgroundHeight : 0,
            child: AnimatedOpacity(
              duration: const Duration(milliseconds: 300),
              opacity: _isEasyMode ? 1.0 : 0.0,
              child: Image.asset(
                "assets/images/easy_background.gif",
                width: screenSize.width,
                fit: BoxFit.cover,
              ),
            ),
          ),
          // 네비게이션 UI (아래로 밀려남)
          AnimatedPositioned(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            top: slideOffset,
            left: 0,
            right: 0,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeInOut,
              height: _isEasyMode ? 120 : 170,
              decoration: BoxDecoration(color: const Color(0xFF2E65C8)),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Row(
                    children: [
                      Container(
                        margin: const EdgeInsets.all(20),
                        width: 50,
                        height: 50,
                        decoration: ShapeDecoration(
                          shape: RoundedRectangleBorder(
                            side: BorderSide(width: 2, color: Colors.white),
                            borderRadius: BorderRadius.circular(100),
                          ),
                        ),
                        child: Center(
                          child: Text(
                            '출발',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 17,
                              fontFamily: 'Pretendard',
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      ),
                      Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '0m',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 39,
                              fontFamily: 'Pretendard',
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          Text(
                            '초량로60번길 방면',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 19,
                              fontFamily: 'Pretendard',
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  SizedBox(height: 20),
                ],
              ),
            ),
          ),
          // 우회전 안내
          AnimatedPositioned(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            top: (_isEasyMode ? 120 : 170) + slideOffset,
            left: 0,
            child: Container(
              width: 170,
              height: 60,
              decoration: ShapeDecoration(
                color: const Color(0xFF2D53A5),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.only(
                    bottomRight: Radius.circular(16),
                  ),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SvgPicture.asset(
                    'assets/icons/turn_right.svg',
                    width: 24,
                    height: 24,
                  ),
                  SizedBox(width: 12),
                  Text(
                    '700m',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 30,
                      fontFamily: 'Pretendard',
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // 속도 아이콘
          AnimatedPositioned(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            top: (_isEasyMode ? 220 : 270) + slideOffset,
            left: 40,
            child: Image.asset(
              'assets/images/speed.png',
              width: 50,
              height: 50,
            ),
          ),
          // 제한속도 아이콘
          AnimatedPositioned(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            top: (_isEasyMode ? 300 : 350) + slideOffset,
            left: 15,
            child: Image.asset(
              'assets/images/limit.png',
              width: 100,
              height: 100,
            ),
          ),
          // Easy 아이콘 (터치 가능, 애니메이션)
          AnimatedPositioned(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            top: _isEasyMode ? (easyBackgroundHeight - iconSize) / 2 : 70.0,
            right: _isEasyMode ? (screenSize.width - iconSize) / 2 : 30.0,
            child: GestureDetector(
              onTap: () => setState(() => _isEasyMode = !_isEasyMode),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeInOut,
                width: iconSize,
                height: iconSize,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(100),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      spreadRadius: 0.1,
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Image.asset('assets/icons/easy.png'),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

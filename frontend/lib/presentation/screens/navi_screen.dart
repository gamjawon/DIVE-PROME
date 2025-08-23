import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class NaviScreen extends StatelessWidget {
  const NaviScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              "assets/images/navi_sample.png",
              fit: BoxFit.cover, // 화면 전체 채우기
            ),
          ),
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 170,
              decoration: BoxDecoration(color: const Color(0xFF2E65C8)),
              child: Column(
                children: [
                  SizedBox(height: 60),
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
                          SizedBox(
                            child: Text(
                              '초량로60번길 방면',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 19,
                                fontFamily: 'Pretendard',
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          Positioned(
            top: 170,
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
          Positioned(
            top: 270,
            left: 40,
            child: Image.asset(
              'assets/images/speed.png',
              width: 50,
              height: 50,
            ),
          ),
          Positioned(
            top: 350,
            left: 15,
            child: Image.asset(
              'assets/images/limit.png',
              width: 100,
              height: 100,
            ),
          ),
        ],
      ),
    );
  }
}

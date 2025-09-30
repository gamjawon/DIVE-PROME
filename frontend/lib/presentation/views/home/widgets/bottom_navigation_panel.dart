import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:frontend/presentation/utils/palette.dart';

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
                color: Palette.primaryAccentColor,
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

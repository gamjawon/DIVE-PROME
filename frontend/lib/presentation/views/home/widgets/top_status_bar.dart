import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:frontend/presentation/viewmodels/home_viewmodel.dart';

class TopStatusBar extends ConsumerWidget {
  const TopStatusBar({super.key, required this.screenWidth});

  final double screenWidth;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final locationState = ref.watch(locationViewmodelProvider);

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
              child: Image.asset(
                'assets/icons/profile.png',
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
                          location == null ? '위치 정보 없음' : location.addressName,
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

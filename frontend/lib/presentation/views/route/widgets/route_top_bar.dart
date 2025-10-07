import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/svg.dart';
import 'package:frontend/presentation/viewmodels/route_viewmodel.dart';

class RouteTopBar extends ConsumerWidget {
  const RouteTopBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final routeStateAsync = ref.watch(routeViewmodelProvider);

    return Positioned(
      top: 70,
      left: 40,
      right: 40,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: SvgPicture.asset(
              'assets/icons/back.svg',
              width: 30,
              height: 30,
            ),
          ),
          Container(
            width: 300,
            height: 60,
            decoration: ShapeDecoration(
              color: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15),
              ),
              shadows: [
                BoxShadow(
                  color: Color(0x3F9F9F9F),
                  blurRadius: 3.84,
                  offset: Offset(-0.96, 0),
                  spreadRadius: 0,
                ),
                BoxShadow(
                  color: Color(0x3F787878),
                  blurRadius: 3.84,
                  offset: Offset(0.96, 0.96),
                  spreadRadius: 0,
                ),
              ],
            ),
            child: routeStateAsync.when(
              data: (routeState) => Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Expanded(
                    child: Text(
                      routeState.start?.placeName ?? '출발지',
                      textAlign: TextAlign.center,
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                      style: TextStyle(
                        color: const Color(0xFF374151),
                        fontSize: 18,
                        fontFamily: 'Pretendard',
                        fontWeight: FontWeight.w500,
                        height: 1.29,
                      ),
                    ),
                  ),
                  Container(
                    margin: EdgeInsets.symmetric(horizontal: 10),
                    child: SvgPicture.asset(
                      'assets/icons/arrow.svg',
                      width: 16,
                      height: 16,
                    ),
                  ),
                  Expanded(
                    child: Text(
                      routeState.end?.placeName ?? '도착지',
                      textAlign: TextAlign.center,
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                      style: TextStyle(
                        color: const Color(0xFF374151),
                        fontSize: 18,
                        fontFamily: 'Pretendard',
                        fontWeight: FontWeight.w500,
                        height: 1.29,
                      ),
                    ),
                  ),
                ],
              ),
              loading: () => Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Expanded(child: Text('출발지', textAlign: TextAlign.center)),
                  Container(
                    margin: EdgeInsets.symmetric(horizontal: 10),
                    child: SvgPicture.asset(
                      'assets/icons/arrow.svg',
                      width: 16,
                      height: 16,
                    ),
                  ),
                  Expanded(child: Text('도착지', textAlign: TextAlign.center)),
                ],
              ),
              error: (_, __) => Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Expanded(child: Text('출발지', textAlign: TextAlign.center)),
                  Container(
                    margin: EdgeInsets.symmetric(horizontal: 10),
                    child: SvgPicture.asset(
                      'assets/icons/arrow.svg',
                      width: 16,
                      height: 16,
                    ),
                  ),
                  Expanded(child: Text('도착지', textAlign: TextAlign.center)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

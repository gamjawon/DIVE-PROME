import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/svg.dart';
import 'package:frontend/data/models/route_model.dart';
import 'package:frontend/presentation/states/route_state.dart';
import 'package:frontend/presentation/utils/palette.dart';
import 'package:frontend/presentation/viewmodels/route_viewmodel.dart';
import 'package:frontend/presentation/views/navigation/navigation_screen.dart';

class RouteBottomContainer extends ConsumerWidget {
  const RouteBottomContainer({super.key});

  List<Widget> _buildRouteTags(RouteOption option) {
    List<String> tags = [];

    switch (option) {
      case RouteOption.recommend:
        tags = ['거리 우선', '시간 우선'];
        break;
      case RouteOption.mainRoad:
        tags = ['넓은폭', '주요도로'];
        break;
      case RouteOption.easy:
        tags = ['적은 정체량', '넓은폭', '편한길'];
        break;
    }

    List<Widget> tagWidgets = [];
    for (int i = 0; i < tags.length; i++) {
      if (i > 0) {
        tagWidgets.add(SizedBox(width: 8));
      }
      tagWidgets.add(_buildRouteTag(tags[i]));
    }

    return tagWidgets;
  }

  Widget _buildRouteTag(String label) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Color(0xFFEFEFF0),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Color(0xFFD1D5DB), width: 1),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w500,
          color: Color(0xFF6B7280),
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, Color routeColor) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 20,
            color: Colors.black45,
            fontWeight: FontWeight.w500,
          ),
        ),
        SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 20,
            color: routeColor,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final routeStateAsync = ref.watch(routeViewmodelProvider);

    return routeStateAsync.when(
      data: (routeState) => _buildContent(context, ref, routeState),
      loading: () => SizedBox.shrink(),
      error: (_, __) => SizedBox.shrink(),
    );
  }

  Widget _buildContent(
    BuildContext context,
    WidgetRef ref,
    RouteState routeState,
  ) {
    if (routeState.routeList == null) return SizedBox.shrink();

    // routes는 이제 Map<String, RouteInfo> 형태입니다
    final routeList = routeState.routeList!;
    if (routeList.isEmpty) return SizedBox.shrink();

    final selectedRoute = routeList.firstWhere(
      (route) => route.option == routeState.selectedOption,
      orElse: () => routeList.first,
    );

    return Positioned(
      bottom: 20,
      left: 0,
      right: 0,
      child: Container(
        decoration: ShapeDecoration(
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.only(
              topLeft: Radius.circular(23.06),
              topRight: Radius.circular(23.06),
            ),
          ),
        ),
        clipBehavior: Clip.antiAlias,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // 경로 옵션 탭
            SizedBox(
              height: 76,
              child: Row(
                children: RouteOption.values.map((option) {
                  final isSelected = option == routeState.selectedOption;
                  final optionColor = Palette.routeColors[option]!;
                  return Expanded(
                    child: GestureDetector(
                      onTap: () {
                        ref
                            .read(routeViewmodelProvider.notifier)
                            .setSelectedOption(option);
                      },
                      child: Container(
                        decoration: BoxDecoration(
                          color: isSelected
                              ? optionColor.withValues(alpha: 0.1)
                              : Colors.white,
                          borderRadius: BorderRadius.only(
                            topLeft: Radius.circular(
                              option == RouteOption.easy ? 23.06 : 0,
                            ),
                            topRight: Radius.circular(
                              option == RouteOption.mainRoad ? 23.06 : 0,
                            ),
                          ),
                        ),
                        child: Center(
                          child: Text(
                            option.displayName,
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: isSelected
                                  ? FontWeight.w600
                                  : FontWeight.w400,
                              color: isSelected
                                  ? optionColor
                                  : Color(0xFF6B7280),
                            ),
                          ),
                        ),
                      ),
                    ),
                  );
                }).toList(),
              ),
            ),
            // 선택된 경로 정보
            Container(
              padding: const EdgeInsets.fromLTRB(32, 16, 32, 36),
              decoration: BoxDecoration(color: Colors.white),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: 8),
                  // 시간과 거리, 안내시작 버튼
                  Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // 시간과 거리
                            Row(
                              children: [
                                Text(
                                  '${selectedRoute.durationMin}분',
                                  style: TextStyle(
                                    fontSize: 40,
                                    fontWeight: FontWeight.w700,
                                    color: Color(0xFF111827),
                                    height: 1.0,
                                  ),
                                ),
                                Container(
                                  margin: EdgeInsets.symmetric(horizontal: 16),
                                  height: 30,
                                  decoration: ShapeDecoration(
                                    shape: RoundedRectangleBorder(
                                      side: BorderSide(
                                        width: 1,
                                        strokeAlign:
                                            BorderSide.strokeAlignCenter,
                                        color: const Color(0xFFD1D5DB),
                                      ),
                                    ),
                                  ),
                                ),
                                Text(
                                  '${selectedRoute.distanceKm.toStringAsFixed(1)}km',
                                  style: TextStyle(
                                    fontSize: 24,
                                    fontWeight: FontWeight.w400,
                                    color: Color(0xFF6B7280),
                                  ),
                                ),
                              ],
                            ),
                            SizedBox(height: 24),
                            // 경로 특성 태그
                            Row(
                              children: _buildRouteTags(selectedRoute.option),
                            ),
                          ],
                        ),
                      ),
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => NavigationScreen(),
                            ),
                          );
                        },
                        child: Column(
                          children: [
                            Container(
                              width: 80,
                              height: 80,
                              padding: EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 12,
                              ),
                              decoration: BoxDecoration(
                                color: Palette
                                    .routeColors[routeState.selectedOption]!,
                                borderRadius: BorderRadius.circular(100),
                              ),
                              child: SvgPicture.asset(
                                'assets/icons/start_navi.svg',
                                width: 30,
                                height: 30,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              '안내시작',
                              style: TextStyle(
                                color: Palette
                                    .routeColors[routeState.selectedOption]!,
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  Container(
                    margin: EdgeInsets.symmetric(vertical: 32),
                    width: MediaQuery.sizeOf(context).width - 32,
                    height: 1,
                    color: const Color(0xFFC9C9C9),
                  ),
                  // 상세 통계
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _buildStatItem(
                        '차선 변경',
                        '${selectedRoute.laneChanges}회',
                        Palette.routeColors[routeState.selectedOption]!,
                      ),
                      _buildStatItem(
                        'U턴 횟수',
                        '${selectedRoute.uTurns}회',
                        Palette.routeColors[routeState.selectedOption]!,
                      ),
                      _buildStatItem(
                        '급경사로 수',
                        '${selectedRoute.steepSlopes}회',
                        Palette.routeColors[routeState.selectedOption]!,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

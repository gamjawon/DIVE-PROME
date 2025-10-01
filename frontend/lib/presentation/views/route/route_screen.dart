import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/presentation/viewmodels/route_viewmodel.dart';
import 'package:frontend/presentation/views/route/widgets/route_bottom_container.dart';
import 'package:frontend/presentation/views/route/widgets/route_map.dart';
import 'package:frontend/presentation/views/route/widgets/route_top_bar.dart';

class RouteScreen extends ConsumerStatefulWidget {
  const RouteScreen({super.key});

  @override
  ConsumerState<RouteScreen> createState() => _RouteViewScreenState();
}

class _RouteViewScreenState extends ConsumerState<RouteScreen> {
  @override
  Widget build(BuildContext context) {
    final routeState = ref.watch(routeViewmodelProvider);

    return Scaffold(
      body: Stack(
        children: [
          routeState.when(
            data: (state) {
              if (state.routeList == null || state.routeList!.isEmpty) {
                return const Center(child: Text('경로 데이터가 없습니다.'));
              }

              return RouteMap();
            },
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (error, stack) => Center(child: Text('오류: $error')),
          ),
          RouteTopBar(),
          // 하단 여백을 채우는 컨테이너
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            height: 20,
            child: Container(color: Colors.white),
          ),
          RouteBottomContainer(),
        ],
      ),
    );
  }
}

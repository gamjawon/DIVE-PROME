//==============**UI 화면** (출발/도착 입력 → 대안 경로 보여주기).=============

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../state/nav_state.dart';
import '../services/directions_service.dart';

class NavPage extends ConsumerStatefulWidget {
  const NavPage({super.key});

  @override
  ConsumerState<NavPage> createState() => _NavPageState();
}

class _NavPageState extends ConsumerState<NavPage> {
  final TextEditingController originCtrl = TextEditingController();
  final TextEditingController destCtrl = TextEditingController();
  late DirectionsService directions;

  @override
  void initState() {
    super.initState();
    directions = DirectionsService(); // 서비스 인스턴스
  }

  Future<void> _search() async {
    final routes = await directions.fetchEasyRoutes(
      originCtrl.text,
      destCtrl.text,
    );
    ref.read(navProvider.notifier).setOD(originCtrl.text, destCtrl.text);
    ref.read(navProvider.notifier).setCandidates(routes);
  }

  @override
  Widget build(BuildContext context) {
    final nav = ref.watch(navProvider);

    return Scaffold(
      appBar: AppBar(title: const Text("쉬운 내비")),
      body: Column(
        children: [
          // 출발/도착 입력
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: originCtrl,
                    decoration: const InputDecoration(hintText: "출발지"),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: TextField(
                    controller: destCtrl,
                    decoration: const InputDecoration(hintText: "도착지"),
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: _search,
                  child: const Text("검색"),
                )
              ],
            ),
          ),
          const Divider(),
          // 대안 경로 카드
          Expanded(
            child: ListView.builder(
              itemCount: nav.candidates.length,
              itemBuilder: (context, i) {
                final r = nav.candidates[i];
                return Card(
                  child: ListTile(
                    title: Text(
                        "${r.id} · ${r.distanceKm.toStringAsFixed(1)}km · ${r.eta.inMinutes}분"),
                    subtitle: Text(
                        "난이도 점수 ${r.difficulty.toStringAsFixed(2)} (낮을수록 쉬움)"),
                    onTap: () =>
                        ref.read(navProvider.notifier).select(r), // 선택 갱신
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

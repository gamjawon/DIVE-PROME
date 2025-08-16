// ================경로 후보 + 난이도 점수 계산===================

class RouteOption {
  final String id;
  final double distanceKm;
  final Duration eta;
  final double difficulty; // 낮을수록 쉬움

  const RouteOption({
    required this.id,
    required this.distanceKm,
    required this.eta,
    required this.difficulty,
  });
}

class DirectionsService {
  // 해커톤: 실제 API 대신 mock 데이터 리턴
  Future<List<RouteOption>> fetchEasyRoutes(String origin, String dest) async {
    await Future.delayed(const Duration(milliseconds: 500)); // 로딩 흉내

    final mock = [
      RouteOption(
        id: "빠른 경로",
        distanceKm: 12.3,
        eta: const Duration(minutes: 25),
        difficulty: _score(curvature: 0.6, slope: 0.5, leftTurns: 5),
      ),
      RouteOption(
        id: "쉬운 경로",
        distanceKm: 13.1,
        eta: const Duration(minutes: 28),
        difficulty: _score(curvature: 0.2, slope: 0.3, leftTurns: 2),
      ),
      RouteOption(
        id: "경치 좋은 경로",
        distanceKm: 14.4,
        eta: const Duration(minutes: 30),
        difficulty: _score(curvature: 0.4, slope: 0.4, leftTurns: 3),
      ),
    ];

    mock.sort((a, b) => a.difficulty.compareTo(b.difficulty)); // 난이도 낮은 순
    return mock;
  }

  double _score({required double curvature, required double slope, required int leftTurns}) {
    // 간단한 가중치
    return curvature * 0.4 + slope * 0.4 + (leftTurns / 10.0) * 0.2;
  }
}

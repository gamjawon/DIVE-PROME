enum RouteOption {
  easy('EASY', '쉬운 길 추천'),
  recommend('RECOMMEND', '내비 추천'),
  mainRoad('MAIN_ROAD', '큰길 우선');

  const RouteOption(this.label, this.displayName);
  final String label;
  final String displayName;

  static RouteOption fromValue(String label) {
    return RouteOption.values.firstWhere(
      (option) => option.label == label,
      orElse: () => RouteOption.easy,
    );
  }
}

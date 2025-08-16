//============경로 상태관리=============

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/directions_service.dart';

class NavState {
  final String? origin;
  final String? dest;
  final List<RouteOption> candidates;
  final RouteOption? selected;

  const NavState({
    this.origin,
    this.dest,
    this.candidates = const [],
    this.selected,
  });

  NavState copyWith({
    String? origin,
    String? dest,
    List<RouteOption>? candidates,
    RouteOption? selected,
  }) {
    return NavState(
      origin: origin ?? this.origin,
      dest: dest ?? this.dest,
      candidates: candidates ?? this.candidates,
      selected: selected ?? this.selected,
    );
  }
}

class NavNotifier extends StateNotifier<NavState> {
  NavNotifier() : super(const NavState());

  void setOD(String o, String d) =>
      state = state.copyWith(origin: o, dest: d);

  void setCandidates(List<RouteOption> list) =>
      state = state.copyWith(candidates: list, selected: list.isNotEmpty ? list.first : null);

  void select(RouteOption r) => state = state.copyWith(selected: r);
}

final navProvider = StateNotifierProvider<NavNotifier, NavState>(
  (ref) => NavNotifier(),
);

import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'navigation_viewmodel.g.dart';

@riverpod
class NavigationViewmodel extends _$NavigationViewmodel {
  @override
  bool build() {
    return false; // 초기 상태: 이지 모드 비활성화
  }

  /// 이지 모드 토글
  void toggleEasyMode() {
    state = !state;
  }

  /// 이지 모드 설정
  void setEasyMode(bool isEasy) {
    state = isEasy;
  }
}

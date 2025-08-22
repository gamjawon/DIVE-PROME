import 'package:flutter/material.dart';
import 'package:frontend/kakao_navi_controller.dart';
import 'package:frontend/kakao_navi_view.dart';
import 'package:permission_handler/permission_handler.dart';

class NaviScreen extends StatefulWidget {
  final double? startLat;
  final double? startLng;
  final double? goalLat;
  final double? goalLng;
  final String? startName;
  final String? goalName;

  const NaviScreen({
    super.key,
    this.startLat,
    this.startLng,
    this.goalLat,
    this.goalLng,
    this.startName,
    this.goalName,
  });

  @override
  State<NaviScreen> createState() => _NaviScreenState();
}

class _NaviScreenState extends State<NaviScreen> {
  bool _isLoading = true;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    _checkPermission();
  }

  Future<void> _checkPermission() async {
    final status = await Permission.location.request();
    if (!status.isGranted) {
      setState(() {
        _isLoading = false;
        _errorMessage = '위치 권한이 필요합니다.\n설정에서 위치 권한을 허용해주세요.';
      });
    }
  }

  Future<void> _onNaviViewCreated(KakaoNaviController controller) async {
    if (!mounted) return;

    try {
      // 자동 인증 및 네비게이션 시작
      await controller.authenticate();

      final startLat = widget.startLat ?? 37.566826;
      final startLng = widget.startLng ?? 126.9786567;
      final goalLat = widget.goalLat ?? 37.4979502;
      final goalLng = widget.goalLng ?? 127.0276368;
      final startName = widget.startName ?? "출발지";
      final goalName = widget.goalName ?? "도착지";

      await controller.startNavigation(
        startLat: startLat,
        startLng: startLng,
        goalLat: goalLat,
        goalLng: goalLng,
        startName: startName,
        goalName: goalName,
      );

      // 네비게이션이 시작되면 현재 화면을 닫고 홈으로 돌아감
      if (mounted) {
        Navigator.of(context).popUntil((route) => route.isFirst);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _errorMessage = '네비게이션 시작에 실패했습니다.\n다시 시도해주세요.';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: true,
      onPopInvokedWithResult: (didPop, result) {
        // 뒤로가기 시 HomeScreen으로 완전히 돌아감
        if (didPop) {
          Navigator.of(context).popUntil((route) => route.isFirst);
        }
      },
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(
          children: [
            // 카카오 내비 뷰
            KakaoNaviView(onViewCreated: _onNaviViewCreated),
            // 로딩 오버레이
            if (_isLoading)
              Container(
                color: Colors.white,
                child: const Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(color: Color(0xFFFF5930)),
                      SizedBox(height: 24),
                      Text(
                        '네비게이션을 시작하고 있습니다...',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                          color: Color(0xFF374151),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            // 에러 오버레이
            if (!_isLoading && _errorMessage.isNotEmpty)
              Container(
                color: Colors.white,
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.error_outline,
                        size: 64,
                        color: Color(0xFFFF5930),
                      ),
                      const SizedBox(height: 24),
                      Text(
                        _errorMessage,
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                          color: Color(0xFF374151),
                        ),
                      ),
                      const SizedBox(height: 32),
                      ElevatedButton(
                        onPressed: () => Navigator.of(context).pop(),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFFF5930),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 32,
                            vertical: 12,
                          ),
                        ),
                        child: const Text(
                          '돌아가기',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

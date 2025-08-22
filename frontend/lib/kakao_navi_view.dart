import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'kakao_navi_controller.dart';

class KakaoNaviView extends StatefulWidget {
  final Function(KakaoNaviController)? onViewCreated;
  final double? width;
  final double? height;

  const KakaoNaviView({super.key, this.onViewCreated, this.width, this.height});

  @override
  State<KakaoNaviView> createState() => _KakaoNaviViewState();
}

class _KakaoNaviViewState extends State<KakaoNaviView> {
  KakaoNaviController? _controller;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.width,
      height: widget.height,
      child: AndroidView(
        viewType: 'kakao_navi_view',
        onPlatformViewCreated: _onPlatformViewCreated,
        creationParamsCodec: const StandardMessageCodec(),
      ),
    );
  }

  void _onPlatformViewCreated(int id) {
    _controller = KakaoNaviController(id);
    widget.onViewCreated?.call(_controller!);
  }
}

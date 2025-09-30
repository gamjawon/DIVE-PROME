import 'package:flutter/material.dart';
import 'package:frontend/presentation/views/home/widgets/bottom_navigation_panel.dart';
import 'package:frontend/presentation/views/home/widgets/kakao_map_background.dart';
import 'package:frontend/presentation/views/home/widgets/route_form.dart';
import 'package:frontend/presentation/views/home/widgets/top_status_bar.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.sizeOf(context).width;

    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Stack(
        children: [
          KakaoMapBackground(),
          Container(width: screenWidth, height: 100, color: Colors.white),
          Positioned(
            top: 0,
            left: 0,
            child: SafeArea(child: TopStatusBar(screenWidth: screenWidth)),
          ),
          Positioned(
            top: 170,
            left: 15,
            right: 15,
            child: RouteForm(screenWidth: screenWidth),
          ),
          Positioned(
            bottom: 20,
            left: 65,
            right: 65,
            child: SafeArea(child: BottomNavigationPanel()),
          ),
        ],
      ),
    );
  }
}

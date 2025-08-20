import 'package:flutter/material.dart';
import 'package:kakao_flutter_sdk/kakao_flutter_sdk.dart';
import 'package:url_launcher/url_launcher.dart';

class NavigationScreen extends StatefulWidget {
  @override
  _NavigationScreenState createState() => _NavigationScreenState();
}

class _NavigationScreenState extends State<NavigationScreen> {
  bool _isNaviInstalled = false;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _checkNaviInstallation();
  }

  // 카카오내비 설치 확인
  Future<void> _checkNaviInstallation() async {
    setState(() {
      _isLoading = true;
    });
    
    try {
      bool isInstalled = await NaviApi.instance.isKakaoNaviInstalled();
      setState(() {
        _isNaviInstalled = isInstalled;
        _isLoading = false;
      });
    } catch (e) {
      print('카카오내비 설치 확인 실패: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  // 카카오 판교오피스로 네비게이션
  Future<void> _navigateToKakao() async {
    setState(() {
      _isLoading = true;
    });

    try {
      if (_isNaviInstalled) {
        await NaviApi.instance.navigate(
          destination: Location(
            name: '카카오 판교오피스',
            x: '127.108640',
            y: '37.402111',
          ),
          option: NaviOption(
            coordType: CoordType.wgs84,
            routeInfo: true,
          ),
        );
        _showMessage('카카오내비로 길안내를 시작합니다.');
      } else {
        await _openKakaoNaviStore();
      }
    } catch (e) {
      _showMessage('네비게이션 실행 실패: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  // 카카오내비 설치 페이지 열기
  Future<void> _openKakaoNaviStore() async {
    const String storeUrl = 'https://play.google.com/store/apps/details?id=com.locnall.KimGiSa';
    
    try {
      final Uri url = Uri.parse(storeUrl);
      
      if (await canLaunchUrl(url)) {
        await launchUrl(url, mode: LaunchMode.externalApplication);
        _showMessage('카카오내비 설치 페이지로 이동합니다.');
      } else {
        _showMessage('앱 스토어를 열 수 없습니다.');
      }
    } catch (e) {
      _showMessage('설치 페이지 이동 실패: $e');
    }
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: Duration(seconds: 3),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('카카오 네비게이션 테스트'),
        backgroundColor: Colors.yellow[700],
        elevation: 0,
      ),
      body: Container(
        padding: EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 상태 표시 카드
            Card(
              elevation: 4,
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  children: [
                    Icon(
                      _isNaviInstalled ? Icons.check_circle : Icons.error,
                      size: 48,
                      color: _isNaviInstalled ? Colors.green : Colors.red,
                    ),
                    SizedBox(height: 12),
                    Text(
                      _isLoading 
                          ? '확인 중...' 
                          : _isNaviInstalled 
                              ? '카카오내비가 설치되어 있습니다'
                              : '카카오내비가 설치되어 있지 않습니다',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: _isNaviInstalled ? Colors.green : Colors.red,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
            
            SizedBox(height: 30),
            
            // 네비게이션 실행 버튼
            ElevatedButton(
              onPressed: _isLoading ? null : _navigateToKakao,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.yellow[700],
                foregroundColor: Colors.black,
                padding: EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: _isLoading
                  ? Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(Colors.black),
                          ),
                        ),
                        SizedBox(width: 12),
                        Text('처리 중...'),
                      ],
                    )
                  : Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.navigation),
                        SizedBox(width: 8),
                        Text(
                          _isNaviInstalled 
                              ? '카카오 판교오피스로 길안내'
                              : '카카오내비 설치하기',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
            ),
            
            SizedBox(height: 16),
            
            // 새로고침 버튼
            TextButton(
              onPressed: _isLoading ? null : _checkNaviInstallation,
              child: Text('설치 상태 다시 확인'),
            ),
            
            SizedBox(height: 30),
            
            // 안내 텍스트
            Container(
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue[200]!),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '📱 사용 방법:',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.blue[800],
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    '1. 카카오내비가 설치되어 있으면 바로 길안내가 시작됩니다.\n'
                    '2. 설치되어 있지 않으면 플레이스토어로 이동합니다.\n'
                    '3. 실제 사용하려면 카카오 API 키를 설정해야 합니다.',
                    style: TextStyle(color: Colors.blue[700]),
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
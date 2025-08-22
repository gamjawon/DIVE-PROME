package com.example.frontend

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    companion object {
        @JvmStatic
        var instance: MainActivity? = null
            private set
    }
    
    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)
        instance = this
    }
    
    override fun onDestroy() {
        super.onDestroy()
        instance = null
    }
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        flutterEngine
            .platformViewsController
            .registry
            .registerViewFactory("kakao_navi_view", KakaoNaviViewFactory(flutterEngine.dartExecutor.binaryMessenger))
    }
}

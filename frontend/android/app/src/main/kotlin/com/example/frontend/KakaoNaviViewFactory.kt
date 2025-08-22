package com.example.frontend

import android.content.Context
import android.graphics.Color
import android.view.View
import android.widget.FrameLayout
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory
import com.kakaomobility.knsdk.ui.view.KNNaviView
import com.kakaomobility.knsdk.common.objects.KNError
import com.kakaomobility.knsdk.common.objects.KNPOI
import com.kakaomobility.knsdk.trip.kntrip.KNTrip
import com.kakaomobility.knsdk.KNRoutePriority
import com.kakaomobility.knsdk.KNLanguageType
import com.kakaomobility.knsdk.guidance.knguidance.KNGuidance
import com.kakaomobility.knsdk.guidance.knguidance.KNGuidance_CitsGuideDelegate
import com.kakaomobility.knsdk.guidance.knguidance.KNGuidance_GuideStateDelegate
import com.kakaomobility.knsdk.guidance.knguidance.KNGuidance_LocationGuideDelegate
import com.kakaomobility.knsdk.guidance.knguidance.KNGuidance_RouteGuideDelegate
import com.kakaomobility.knsdk.guidance.knguidance.KNGuidance_SafetyGuideDelegate
import com.kakaomobility.knsdk.guidance.knguidance.KNGuidance_VoiceGuideDelegate
import com.kakaomobility.knsdk.guidance.knguidance.KNGuideRouteChangeReason
import com.kakaomobility.knsdk.guidance.knguidance.citsguide.KNGuide_Cits
import com.kakaomobility.knsdk.guidance.knguidance.common.KNLocation
import com.kakaomobility.knsdk.guidance.knguidance.locationguide.KNGuide_Location
import com.kakaomobility.knsdk.guidance.knguidance.routeguide.KNGuide_Route
import com.kakaomobility.knsdk.guidance.knguidance.routeguide.objects.KNMultiRouteInfo
import com.kakaomobility.knsdk.guidance.knguidance.safetyguide.KNGuide_Safety
import com.kakaomobility.knsdk.guidance.knguidance.safetyguide.objects.KNSafety
import com.kakaomobility.knsdk.guidance.knguidance.voiceguide.KNGuide_Voice
import com.kakaomobility.knsdk.trip.kntrip.knroute.KNRoute
import android.util.Log
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.BinaryMessenger
import androidx.appcompat.view.ContextThemeWrapper

class KakaoNaviViewFactory(private val messenger: BinaryMessenger) : PlatformViewFactory(StandardMessageCodec.INSTANCE) {
    override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
        // MainActivity 인스턴스를 사용
        val activityContext = MainActivity.instance ?: context
        return KakaoNaviPlatformView(activityContext, viewId, messenger)
    }
}

class KakaoNaviPlatformView(
    context: Context,
    id: Int,
    messenger: BinaryMessenger
) : PlatformView,
    MethodChannel.MethodCallHandler {

    private var naviView: KNNaviView? = null
    private val container: FrameLayout
    private val methodChannel: MethodChannel
    private var isAuthenticated = false

    init {
        container = FrameLayout(context)
        
        // 카카오 내비 초기화 시도
        try {
            // AppCompatActivity가 필요하므로 우선 간단한 뷰로 대체
            Log.i("KakaoNavi", "Creating platform view...")
            
            // 임시로 간단한 텍스트 뷰 생성
            val textView = android.widget.TextView(context)
            textView.text = "네비게이션을 준비하고 있습니다..."
            textView.gravity = android.view.Gravity.CENTER
            textView.setBackgroundColor(android.graphics.Color.WHITE)
            textView.setTextColor(android.graphics.Color.parseColor("#374151"))
            textView.setPadding(50, 50, 50, 50)
            textView.textSize = 16f
            
            container.addView(textView, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            ))
            
            Log.i("KakaoNavi", "Platform view created successfully")
            
        } catch (e: Exception) {
            Log.e("KakaoNavi", "Failed to create platform view: ${e.message}")
            val errorView = android.widget.TextView(context)
            errorView.text = "오류: ${e.message}"
            errorView.gravity = android.view.Gravity.CENTER
            errorView.setBackgroundColor(android.graphics.Color.RED)
            container.addView(errorView)
        }
        
        // Method Channel 설정
        methodChannel = MethodChannel(messenger, "kakao_navi_view_$id")
        methodChannel.setMethodCallHandler(this)
    }

    override fun getView(): View = container

    override fun dispose() {
        methodChannel.setMethodCallHandler(null)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "authenticate" -> {
                val appKey = call.argument<String>("appKey") ?: "9798daa48b0248d144a73bfa2591ce1d"
                val clientVersion = call.argument<String>("clientVersion") ?: "1.0.0"
                val userKey = call.argument<String>("userKey") ?: "testUser"
                authenticate(appKey, clientVersion, userKey, result)
            }
            "startNavigation" -> {
                val startLat = call.argument<Double>("startLat") ?: 37.566826
                val startLng = call.argument<Double>("startLng") ?: 126.9786567
                val goalLat = call.argument<Double>("goalLat") ?: 37.4979502
                val goalLng = call.argument<Double>("goalLng") ?: 127.0276368
                val startName = call.argument<String>("startName") ?: "서울시청"
                val goalName = call.argument<String>("goalName") ?: "강남역"
                startNavigation(startLat, startLng, goalLat, goalLng, startName, goalName, result)
            }
            else -> result.notImplemented()
        }
    }

    private fun authenticate(appKey: String, clientVersion: String, userKey: String, result: MethodChannel.Result) {
        KakaoNaviApplication.knsdk.initializeWithAppKey(
            aAppKey = appKey,
            aClientVersion = clientVersion,
            aUserKey = userKey,
            aLangType = KNLanguageType.KNLanguageType_KOREAN
        ) { err: KNError? ->
            if (err != null) {
                Log.e("KNSDK", "Auth fail code=${err.code} detail=$err")
                result.error("AUTH_ERROR", "인증 실패: ${err.code}", err.toString())
            } else {
                isAuthenticated = true
                Log.i("KNSDK", "Authentication successful")
                
                // 인증 성공 후 KNNaviView 생성 시도
                try {
                    val activity = MainActivity.instance
                    if (activity != null) {
                        // AppCompatActivity 호환 컨텍스트 생성
                        val appCompatContext = androidx.appcompat.view.ContextThemeWrapper(
                            activity, 
                            androidx.appcompat.R.style.Theme_AppCompat_Light
                        )
                        
                        naviView = KNNaviView(appCompatContext)
                        naviView?.let { navi ->
                            // 기존 view 제거
                            container.removeAllViews()
                            
                            // 새로운 내비 뷰 추가
                            container.addView(navi, FrameLayout.LayoutParams(
                                FrameLayout.LayoutParams.MATCH_PARENT,
                                FrameLayout.LayoutParams.MATCH_PARENT
                            ))
                            
                            navi.useDarkMode = true
                            Log.i("KNSDK", "KNNaviView created successfully with ContextThemeWrapper")
                        }
                    } else {
                        Log.w("KNSDK", "MainActivity instance is null")
                    }
                } catch (e: Exception) {
                    Log.e("KNSDK", "Failed to create KNNaviView after auth: ${e.message}")
                    // KNNaviView 생성에 실패해도 인증은 성공
                }
                
                result.success("인증 성공")
            }
        }
    }

    private fun startNavigation(
        startLat: Double, startLng: Double,
        goalLat: Double, goalLng: Double,
        startName: String, goalName: String,
        result: MethodChannel.Result
    ) {
        if (!isAuthenticated) {
            result.error("NOT_AUTHENTICATED", "먼저 인증을 완료해주세요", null)
            return
        }

        try {
            val context = MainActivity.instance ?: container.context
            val intent = android.content.Intent(context, KakaoNaviActivity::class.java).apply {
                putExtra(KakaoNaviActivity.EXTRA_START_LAT, startLat)
                putExtra(KakaoNaviActivity.EXTRA_START_LNG, startLng)
                putExtra(KakaoNaviActivity.EXTRA_GOAL_LAT, goalLat)
                putExtra(KakaoNaviActivity.EXTRA_GOAL_LNG, goalLng)
                putExtra(KakaoNaviActivity.EXTRA_START_NAME, startName)
                putExtra(KakaoNaviActivity.EXTRA_GOAL_NAME, goalName)
                flags = android.content.Intent.FLAG_ACTIVITY_NEW_TASK
            }
            
            context.startActivity(intent)
            result.success("내비게이션 시작")
            
        } catch (e: Exception) {
            Log.e("KNSDK", "Failed to start navigation activity: ${e.message}")
            result.error("NAVIGATION_ERROR", "내비게이션 시작 실패: ${e.message}", e.toString())
        }
    }
}

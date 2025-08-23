package com.example.frontend

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import com.kakaomobility.knsdk.ui.view.KNNaviView
import com.kakaomobility.knsdk.KNLanguageType
import com.kakaomobility.knsdk.KNRoutePriority
import com.kakaomobility.knsdk.common.objects.KNError
import com.kakaomobility.knsdk.common.objects.KNPOI
import com.kakaomobility.knsdk.trip.kntrip.KNTrip
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

class KakaoNaviActivity : AppCompatActivity(),
    KNGuidance_GuideStateDelegate,
    KNGuidance_LocationGuideDelegate,
    KNGuidance_RouteGuideDelegate,
    KNGuidance_SafetyGuideDelegate,
    KNGuidance_VoiceGuideDelegate,
    KNGuidance_CitsGuideDelegate {

    private var naviView: KNNaviView? = null
    private var container: FrameLayout? = null

    companion object {
        const val EXTRA_START_LAT = "start_lat"
        const val EXTRA_START_LNG = "start_lng"
        const val EXTRA_GOAL_LAT = "goal_lat"
        const val EXTRA_GOAL_LNG = "goal_lng"
        const val EXTRA_START_NAME = "start_name"
        const val EXTRA_GOAL_NAME = "goal_name"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // 전체 화면 설정
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        
        // 시스템 UI 숨기기 (상태바, 네비게이션 바)
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
            or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
            or View.SYSTEM_UI_FLAG_FULLSCREEN
            or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        )
        
        container = FrameLayout(this)
        setContentView(container)

        try {
            // KNNaviView 생성
            naviView = KNNaviView(this)
            naviView?.let { navi ->
                container?.addView(navi, FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT
                ))
                
                navi.useDarkMode = true
                Log.i("KakaoNaviActivity", "KNNaviView created successfully")
                
                // Intent에서 파라미터 받아서 내비게이션 시작
                startNavigationFromIntent()
            }
        } catch (e: Exception) {
            Log.e("KakaoNaviActivity", "Failed to create KNNaviView: ${e.message}")
            finish()
        }
    }

    private fun startNavigationFromIntent() {
        val startLat = intent.getDoubleExtra(EXTRA_START_LAT, 37.566826)
        val startLng = intent.getDoubleExtra(EXTRA_START_LNG, 126.9786567)
        val goalLat = intent.getDoubleExtra(EXTRA_GOAL_LAT, 37.4979502)
        val goalLng = intent.getDoubleExtra(EXTRA_GOAL_LNG, 127.0276368)
        val startName = intent.getStringExtra(EXTRA_START_NAME) ?: "서울시청"
        val goalName = intent.getStringExtra(EXTRA_GOAL_NAME) ?: "강남역"

        Thread {
            // WGS84 좌표를 KATEC 좌표로 변환
            val startKatecX = ((startLng - 126.0) * 200000).toInt() + 200000
            val startKatecY = ((startLat - 37.0) * 200000).toInt() + 450000
            val goalKatecX = ((goalLng - 126.0) * 200000).toInt() + 200000
            val goalKatecY = ((goalLat - 37.0) * 200000).toInt() + 450000

            val startPoi = KNPOI(startName, startKatecX, startKatecY, startName)
            val goalPoi = KNPOI(goalName, goalKatecX, goalKatecY, goalName)

            KakaoNaviApplication.knsdk.makeTripWithStart(
                aStart = startPoi,
                aGoal = goalPoi,
                aVias = null
            ) { err: KNError?, trip: KNTrip? ->
                if (err == null && trip != null) {
                    runOnUiThread {
                        startGuide(trip)
                    }
                } else {
                    Log.e("KakaoNaviActivity", "Route fail: ${err?.code} detail=$err")
                    runOnUiThread {
                        finish()
                    }
                }
            }
        }.start()
    }

    private fun startGuide(trip: KNTrip) {
        naviView?.let { navi ->
            KakaoNaviApplication.knsdk.sharedGuidance()?.apply {
                guideStateDelegate = this@KakaoNaviActivity
                locationGuideDelegate = this@KakaoNaviActivity
                routeGuideDelegate = this@KakaoNaviActivity
                safetyGuideDelegate = this@KakaoNaviActivity
                voiceGuideDelegate = this@KakaoNaviActivity
                citsGuideDelegate = this@KakaoNaviActivity

                navi.initWithGuidance(
                    this,
                    trip,
                    KNRoutePriority.KNRoutePriority_Recommand,
                    0
                )
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        naviView = null
        container = null
    }

    // Delegate 메소드들
    override fun guidanceGuideStarted(aGuidance: KNGuidance) { 
        naviView?.guidanceGuideStarted(aGuidance) 
    }
    
    override fun guidanceCheckingRouteChange(aGuidance: KNGuidance) { 
        naviView?.guidanceCheckingRouteChange(aGuidance) 
    }
    
    override fun guidanceRouteUnchanged(aGuidance: KNGuidance) { 
        naviView?.guidanceRouteUnchanged(aGuidance) 
    }
    
    override fun guidanceRouteUnchangedWithError(aGuidnace: KNGuidance, aError: KNError) { 
        naviView?.guidanceRouteUnchangedWithError(aGuidnace, aError) 
    }
    
    override fun guidanceOutOfRoute(aGuidance: KNGuidance) { 
        naviView?.guidanceOutOfRoute(aGuidance) 
    }
    
    override fun guidanceRouteChanged(aGuidance: KNGuidance, aFromRoute: KNRoute, aFromLocation: KNLocation, aToRoute: KNRoute, aToLocation: KNLocation, aChangeReason: KNGuideRouteChangeReason) {
        naviView?.guidanceRouteChanged(aGuidance)
    }
    
    override fun guidanceGuideEnded(aGuidance: KNGuidance) { 
        naviView?.guidanceGuideEnded(aGuidance) 
        finish() // 내비게이션 종료 시 액티비티도 종료
    }
    
    override fun guidanceDidUpdateLocation(aGuidance: KNGuidance, aLocationGuide: KNGuide_Location) { 
        naviView?.guidanceDidUpdateLocation(aGuidance, aLocationGuide) 
    }
    
    override fun guidanceDidUpdateRouteGuide(aGuidance: KNGuidance, aRouteGuide: KNGuide_Route) { 
        naviView?.guidanceDidUpdateRouteGuide(aGuidance, aRouteGuide) 
    }
    
    override fun guidanceDidUpdateRoutes(aGuidance: KNGuidance, aRoutes: List<KNRoute>, aMultiRouteInfo: KNMultiRouteInfo?) { 
        naviView?.guidanceDidUpdateRoutes(aGuidance, aRoutes, aMultiRouteInfo) 
    }
    
    override fun guidanceDidUpdateSafetyGuide(aGuidance: KNGuidance, aSafetyGuide: KNGuide_Safety?) { 
        naviView?.guidanceDidUpdateSafetyGuide(aGuidance, aSafetyGuide) 
    }
    
    override fun guidanceDidUpdateAroundSafeties(aGuidance: KNGuidance, aSafeties: List<KNSafety>?) { 
        naviView?.guidanceDidUpdateAroundSafeties(aGuidance, aSafeties) 
    }
    
    override fun shouldPlayVoiceGuide(aGuidance: KNGuidance, aVoiceGuide: KNGuide_Voice, aNewData: MutableList<ByteArray>): Boolean =
        naviView?.shouldPlayVoiceGuide(aGuidance, aVoiceGuide, aNewData) ?: false
    
    override fun willPlayVoiceGuide(aGuidance: KNGuidance, aVoiceGuide: KNGuide_Voice) { 
        naviView?.willPlayVoiceGuide(aGuidance, aVoiceGuide) 
    }
    
    override fun didFinishPlayVoiceGuide(aGuidance: KNGuidance, aVoiceGuide: KNGuide_Voice) { 
        naviView?.didFinishPlayVoiceGuide(aGuidance, aVoiceGuide) 
    }
    
    override fun didUpdateCitsGuide(aGuidance: KNGuidance, aCitsGuide: KNGuide_Cits) { 
        naviView?.didUpdateCitsGuide(aGuidance, aCitsGuide) 
    }
}

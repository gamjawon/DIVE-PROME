package com.example.frontend

import android.app.Application
import com.kakaomobility.knsdk.KNSDK
import java.io.File

class KakaoNaviApplication : Application() {
    companion object {
        lateinit var knsdk: KNSDK
            private set
    }

    override fun onCreate() {
        super.onCreate()
        knsdk = KNSDK.apply {
            val root = File(filesDir, "KNSample").absolutePath
            install(this@KakaoNaviApplication, root)
        }
    }
}

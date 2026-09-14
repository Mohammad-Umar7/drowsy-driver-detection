package com.umar.drowsy

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

/**
 * Loads OpenCV's native library before ANY Activity object exists.
 *
 * Application.onCreate runs before the first Activity is even constructed,
 * so by the time a field initializer such as `private val mat = Mat()` runs,
 * the JNI symbols it needs are already there.
 *
 * Why this matters: the load used to happen in MainActivity.onCreate, which
 * runs AFTER the Activity's fields are initialised. Every OpenCV object held
 * as a field - a Mat, a CLAHE - therefore called into native code that was
 * not loaded yet, and the app died on launch with UnsatisfiedLinkError. The
 * build was green throughout; only running it showed it. Loading here makes
 * the order a guarantee rather than something every field has to remember.
 */
class DrowsyApp : Application() {

    override fun onCreate() {
        super.onCreate()
        opencvReady = OpenCVLoader.initLocal()
        if (!opencvReady) Log.e("Drowsy", "OpenCVLoader.initLocal() returned false")
    }

    companion object {
        /** False if the native library could not be loaded on this device. */
        @Volatile var opencvReady = false
            private set
    }
}

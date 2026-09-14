plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.umar.drowsy"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.umar.drowsy"
        // 24 = Android 7.0. MediaPipe Tasks requires 24, and it covers
        // essentially every phone still in use.
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            // OpenCV and ONNX Runtime ship native .so libraries for several
            // CPU architectures. Bundling all of them roughly triples the APK
            // for code that can never run on this phone. arm64-v8a covers
            // every phone since ~2017; armeabi-v7a is kept for older devices.
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    buildTypes {
        release {
            // Left off so the first build is simple and stack traces are
            // readable. Turning this on shrinks the APK considerably but needs
            // keep-rules for ONNX Runtime and MediaPipe reflection.
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"),
                          "proguard-rules.pro")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    androidResources {
        // Do not let AAPT compress these. The .task is already a zip and the
        // .onnx is already compact, so compressing again wastes build time and
        // forces a decompress step at load.
        noCompress += listOf("task", "onnx")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }

    splits {
        // One universal APK carrying both architectures is ~105 MB, because
        // OpenCV, ONNX Runtime and MediaPipe each ship a large .so per ABI.
        // Splitting produces a per-architecture APK roughly half that size,
        // while still emitting the universal one for anyone unsure which
        // their phone needs.
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "armeabi-v7a")
            isUniversalApk = true
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    // CameraX: handles the camera lifecycle, rotation and frame delivery.
    // Doing this with the raw Camera2 API is possible but is hundreds of lines
    // of state machine that CameraX has already debugged across every vendor.
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // The same FaceMesh the desktop version uses, as an on-device library.
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    // Runs our exported eyenet.onnx. No network, no server.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    // The reason this is here: preprocess.py applies CLAHE to the eye crop
    // before the CNN sees it, and training/serving must preprocess IDENTICALLY
    // or accuracy quietly collapses. Using the same OpenCV implementation on
    // the phone removes that whole class of bug, instead of hand-rolling a
    // CLAHE in Kotlin and hoping it matches.
    implementation("org.opencv:opencv:4.12.0")

    // Plain JVM tests for the temporal state machine. Drowsiness.kt has no
    // Android dependency, so it runs on the desktop JVM in seconds:
    //     ./gradlew testDebugUnitTest
    testImplementation("junit:junit:4.13.2")
}

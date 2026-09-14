package com.umar.drowsy

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.umar.drowsy.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/**
 * The whole app in one screen.
 *
 *     camera frame -> MediaPipe landmarks -> eye crops -> ONNX -> monitor -> HUD
 *
 * Exactly the desktop pipeline from src/infer.py, running on the phone with no
 * network involved at any point.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService

    private var landmarker: FaceLandmarker? = null
    private var classifier: EyeClassifier? = null
    private var alarm: Alarm? = null
    private val monitor = DrowsinessMonitor()
    private val calibrator = EarCalibrator()

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var earThresh = Cfg.EAR_THRESH

    // Focal length of the current lens as a fraction of the frame's long
    // side, from the real camera characteristics. 0 when unknown, in which
    // case head pose falls back to the desktop's "long side" approximation.
    // Written on the main thread when a camera binds, read per frame.
    @Volatile private var focalRatio = 0.0

    // Rolling FPS. A single-frame estimate is far too noisy to read.
    private val frameTimes = ArrayDeque<Double>()
    private var lastFrameT = 0.0

    // Everything backed by OpenCV native memory is created in onCreate, after
    // the library is confirmed loaded - never as a field initializer.
    //
    // Field initializers run while the Activity object is being constructed,
    // BEFORE onCreate. `Mat()` and `createCLAHE` are native calls, and with
    // the library loaded in onCreate they ran first and threw
    // UnsatisfiedLinkError: the app crashed on every launch. The build was
    // green the whole time; only running it on a device showed it. DrowsyApp
    // now loads OpenCV before any Activity exists, and these stay lateinit so
    // a device where that load fails gets a message instead of a crash.
    private lateinit var lighting: LightingNormalizer

    // Allocated once and reused. Creating a Bitmap or Mat inside a 30 fps loop
    // churns megabytes per second and hands the garbage collector work inside
    // the frame budget, which is what makes camera apps stutter.
    private lateinit var uprightMat: Mat
    private lateinit var grayMat: Mat
    private var uprightBitmap: Bitmap? = null

    // MediaPipe's VIDEO mode requires STRICTLY increasing timestamps. Two
    // frames can land in the same millisecond on a fast device, and passing
    // the same value twice throws.
    private var lastStampMs = 0L

    private val permission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, R.string.camera_needed, Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)

        // Targeting Android 15 makes the app edge-to-edge whether it asks or
        // not: the layout runs under the status bar and the gesture bar. The
        // camera preview is welcome there, but the cards and the buttons are
        // not - the clock sat on top of the status pill and the gesture
        // handle on top of the buttons. Hand the bar heights to the overlay
        // and pad the button row, and leave the preview full-bleed.
        val pad8 = (8 * resources.displayMetrics.density).toInt()
        val pad12 = (12 * resources.displayMetrics.density).toInt()
        ViewCompat.setOnApplyWindowInsetsListener(ui.root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            ui.overlay.setInsets(bars.top.toFloat(), bars.bottom.toFloat())
            ui.buttons.setPadding(pad8, 0, pad8, pad12 + bars.bottom)
            insets
        }

        // DrowsyApp loaded the native library before this Activity existed.
        // If that failed, nothing below can work - every OpenCV call would
        // throw UnsatisfiedLinkError - so leave cleanly instead of crashing.
        if (!DrowsyApp.opencvReady && !OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show()
            Log.e(TAG, "OpenCV native library unavailable")
            finish()
            return
        }
        lighting = LightingNormalizer()
        uprightMat = Mat()
        grayMat = Mat()

        analysisExecutor = Executors.newSingleThreadExecutor()
        alarm = Alarm(this)
        classifier = try {
            EyeClassifier(this)
        } catch (e: Exception) {
            Log.e(TAG, "ONNX model failed to load", e)
            Toast.makeText(this, "Eye model failed to load", Toast.LENGTH_LONG).show()
            null
        }
        setupLandmarker()

        ui.btnCalibrate.setOnClickListener {
            calibrator.start()
            Toast.makeText(this, R.string.toast_calibrating, Toast.LENGTH_SHORT).show()
        }
        ui.btnReset.setOnClickListener {
            // The monitor is owned by the analysis thread, so the reset runs
            // there too rather than racing an in-flight update().
            analysisExecutor.execute { monitor.reset() }
            ui.overlay.clearHistory()
            Toast.makeText(this, R.string.toast_reset, Toast.LENGTH_SHORT).show()
        }
        ui.btnSwitch.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            // The other camera is a different sensor pointed at a different
            // scene: the eased gamma and the rolling FPS window both describe
            // the OLD one and would be wrong for the first second of the new.
            lighting.reset()
            Geometry.resetPoseSeed()
            frameTimes.clear()
            lastFrameT = 0.0
            startCamera()
        }
        ui.btnAlarm.setOnClickListener {
            val on = alarm?.toggle() ?: false
            ui.btnAlarm.setText(if (on) R.string.btn_mute else R.string.btn_unmute)
        }

        // Reuse a saved calibration if one exists; otherwise measure this
        // driver straight away rather than relying on them remembering to
        // press a button. Desktop has saved its threshold to calibration.json
        // since day one; the phone was re-calibrating from scratch on every
        // launch and forgetting the result the moment the app closed.
        val saved = loadEarThresh()
        if (saved != null) {
            earThresh = saved
            monitor.earThresh = saved
            Toast.makeText(this, "Using saved calibration (EAR %.3f)".format(saved),
                Toast.LENGTH_SHORT).show()
        } else {
            calibrator.start()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }

    // ---- calibration persistence --------------------------------------
    // The threshold is a property of the DRIVER's eyes, not of one session.
    private fun prefs() = getSharedPreferences("drowsy", MODE_PRIVATE)

    private fun loadEarThresh(): Double? {
        val v = prefs().getFloat(PREF_EAR, -1f)
        // Sanity-bounded: a corrupted or absurd value must not lock the
        // detector into a state where no real eye can ever trip it.
        return if (v in 0.05f..0.6f) v.toDouble() else null
    }

    private fun saveEarThresh(v: Double) {
        prefs().edit().putFloat(PREF_EAR, v.toFloat()).apply()
    }

    private fun setupLandmarker() {
        try {
            val base = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()
            val opts = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(base)
                // VIDEO mode is synchronous and tracks between frames instead
                // of re-detecting, which is what makes it fast enough.
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)                       // only the driver matters
                .setMinFaceDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setOutputFaceBlendshapes(false)      // unused, costs time
                .build()
            landmarker = FaceLandmarker.createFromOptions(this, opts)
        } catch (e: Exception) {
            Log.e(TAG, "FaceLandmarker failed", e)
            Toast.makeText(this, "Face model failed to load", Toast.LENGTH_LONG).show()
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(ui.preview.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                // Ask for 720p. Left to itself, ImageAnalysis delivers 640x480
                // - CameraX's documented default - which is the exact mode the
                // desktop version had to fight: at 640x480 an eye at driving
                // distance is ~23 px wide, the eyelid is a few pixels tall,
                // and a 2 px landmark error becomes a 30% error in EAR. The
                // phone was quietly running at half the resolution of the
                // desktop it was ported from. Fall back to the closest mode
                // this sensor supports if 1280x720 is not offered.
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(1280, 720),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                            )
                        )
                        .build()
                )
                // The same principle as the desktop VideoSource: process the
                // NEWEST frame and drop anything that queued up behind it.
                // For a real-time safety signal, fresh beats complete.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // RGBA out of the box saves writing a YUV_420_888 converter.
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { proxy -> onFrame(proxy) }

            // Not every device has both cameras: a tablet in a dash mount
            // may have no rear one, and a few cheap phones have no front.
            // Binding to a lens that is not there throws, and the old catch
            // below just logged it - leaving the preview frozen on the last
            // frame of the OTHER camera with no explanation. Check first and
            // stay on the camera that exists.
            var selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            val available = try { provider.hasCamera(selector) } catch (e: Exception) { false }
            if (!available) {
                val other = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                    CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
                Toast.makeText(this, R.string.camera_missing, Toast.LENGTH_SHORT).show()
                lensFacing = other
                selector = CameraSelector.Builder().requireLensFacing(other).build()
            }

            try {
                provider.unbindAll()
                val camera = provider.bindToLifecycle(this, selector, preview, analysis)
                readLensIntrinsics(camera.cameraInfo)
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
                Toast.makeText(this, R.string.camera_failed, Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Focal length in pixels from the lens itself, not from a guess.
     *
     * Head pose needs the camera's focal length in pixels. The desktop has
     * no way to know it and assumes the frame's long side, which is right
     * to within about 25% for a typical webcam. The phone CAN know it:
     * Camera2 reports the lens focal length in millimetres and the sensor's
     * physical size, and the frame's long side spans the sensor's long side,
     * so   focal_px = focal_mm / sensor_long_mm * frame_long_px.
     * A phone front camera is wide (about 3 mm on a 4.5 mm sensor), so the
     * guess was off by a third: every angle it reported was too large.
     */
    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun readLensIntrinsics(info: CameraInfo) {
        focalRatio = try {
            val c2 = Camera2CameraInfo.from(info)
            val fMm = c2.getCameraCharacteristic(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
            val sensor = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val longMm = if (sensor != null) max(sensor.width, sensor.height) else 0f
            if (fMm > 0f && longMm > 0f) fMm.toDouble() / longMm else 0.0
        } catch (e: Exception) {
            0.0
        }
        Log.i(TAG, "lens focal = %.3f x frame long side".format(focalRatio))
    }

    private fun onFrame(proxy: ImageProxy) {
        val now = System.nanoTime() / 1e9
        try {
            // RGBA frame -> upright Mat, enhanced, then handed to MediaPipe.
            if (!proxy.intoUprightMat(uprightMat)) return
            val w = uprightMat.cols()
            val h = uprightMat.rows()

            // Lighting correction runs FIRST, on the whole frame. Enhancing
            // only the eye crop would be too late: in the dark MediaPipe finds
            // no face, so there is no crop to enhance.
            lighting.process(uprightMat)

            var bmp = uprightBitmap
            if (bmp == null || bmp.width != w || bmp.height != h) {
                bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                uprightBitmap = bmp
            }
            Utils.matToBitmap(uprightMat, bmp)

            var stamp = (now * 1000).toLong()
            if (stamp <= lastStampMs) stamp = lastStampMs + 1
            lastStampMs = stamp

            val result = landmarker?.detectForVideo(
                BitmapImageBuilder(bmp).build(), stamp
            )
            val faces = result?.faceLandmarks()
            val found = !faces.isNullOrEmpty()
            // The next face to appear may be a different person or a very
            // different pose; seeding the pose solver from a stale answer
            // would start it in the wrong basin.
            if (!found) Geometry.resetPoseSeed()

            var ear = 0.3; var mar = 0.0
            var pitch = 0.0; var yaw = 0.0
            var wMax = 0.0; var wMin = 0.0
            var useL = false; var useR = false
            var reliable = false
            var closedProb: Double? = null
            val boxes = mutableListOf<RectF>()

            if (found) {
                val pts = Geometry.toPixels(faces!![0], w, h)

                val earL = Geometry.eyeAspectRatio(pts, Geometry.LEFT_EYE_EAR)
                val earR = Geometry.eyeAspectRatio(pts, Geometry.RIGHT_EYE_EAR)
                mar = Geometry.mouthAspectRatio(pts)
                val ratio = focalRatio
                val focalPx = if (ratio > 0.0) ratio * max(w, h) else max(w, h).toDouble()
                val pose = Geometry.headPose(pts, w, h, focalPx)
                pitch = pose[0]; yaw = pose[1]

                // Which eye can we actually see? On a head turn the far eye is
                // foreshortened to a sliver and both EAR and the CNN read
                // nonsense from it. Measured on the desktop version: at yaw
                // +70 the far eye was 4.6 px wide and scored 0.68 "closed"
                // while the near eye correctly read 0.03.
                val wl = Geometry.eyeWidth(pts, Geometry.LEFT_EYE_EAR)
                val wr = Geometry.eyeWidth(pts, Geometry.RIGHT_EYE_EAR)
                val wm = max(max(wl, wr), 1e-6)
                useL = (wl / wm >= Cfg.EYE_VIS_MIN_RATIO) && wl >= Cfg.EYE_MIN_WIDTH_PX
                useR = (wr / wm >= Cfg.EYE_VIS_MIN_RATIO) && wr >= Cfg.EYE_MIN_WIDTH_PX
                reliable = useL || useR
                wMax = max(wl, wr); wMin = min(wl, wr)

                ear = when {
                    useL && useR -> (earL + earR) / 2.0
                    useL -> earL
                    useR -> earR
                    else -> max(earL, earR)      // nothing trusted; bias open
                }

                val boxL = Geometry.squareBox(pts, Geometry.LEFT_EYE_CONTOUR, 1.35)
                val boxR = Geometry.squareBox(pts, Geometry.RIGHT_EYE_CONTOUR, 1.35)

                // Grayscale comes from the ENHANCED frame, matching the desktop
                // order where preprocess runs after lighting correction.
                Imgproc.cvtColor(uprightMat, grayMat, Imgproc.COLOR_RGBA2GRAY)
                classifier?.let { c ->
                    val p = c.classify(grayMat, boxL, boxR)
                    val usable = mutableListOf<Double>()
                    if (useL) usable.add(p[0])
                    if (useR) usable.add(p[1])
                    closedProb = usable.maxOrNull()
                }

                // Only feed the calibrator when the eyes are actually readable.
                // With the head turned, `ear` is the "nothing trustworthy"
                // fallback, and a value like the 1.05 measured from a 4 px eye
                // would poison the median and set a threshold no real eye
                // could ever fall below - the detector could never fire again.
                if (reliable) calibrator.feed(ear, now)?.let {
                    earThresh = it
                    monitor.earThresh = it
                    saveEarThresh(it)
                    runOnUiThread {
                        Toast.makeText(this,
                            "Calibrated: EAR threshold %.3f".format(it),
                            Toast.LENGTH_SHORT).show()
                    }
                }

                boxes += mapBox(boxL, w, h)
                boxes += mapBox(boxR, w, h)
            }

            val st = monitor.update(
                faceFound = found, closedProb = closedProb, ear = ear, mar = mar,
                pitch = pitch, yaw = yaw, eyesReliable = reliable, now = now
            )
            if (st.shouldAlarm) alarm?.fire(st.alarmKind)

            if (lastFrameT > 0) {
                frameTimes.addLast(now - lastFrameT)
                while (frameTimes.size > 30) frameTimes.removeFirst()
            }
            lastFrameT = now
            val fps = if (frameTimes.isEmpty()) 0.0
                      else 1.0 / max(1e-6, frameTimes.average())

            // One immutable snapshot per frame for the overlay. Built here on
            // the analysis thread, consumed on the UI thread; nothing mutable
            // is shared between the two.
            val frame = HudFrame(
                state = st, faceFound = found, ear = ear, mar = mar,
                pitch = pitch, yaw = yaw, earThresh = earThresh,
                widthMax = wMax, widthMin = wMin, useL = useL, useR = useR,
                fps = fps, boxes = boxes.toList(),
                calibrating = calibrator.armed,
                calibRemaining = calibrator.remaining(now),
                calibTotal = calibrator.seconds,
                light = lighting.stats, lightingOn = lighting.enabled,
                alarmOn = alarm?.enabled ?: false, modelOn = classifier != null,
                now = now
            )
            ui.overlay.post { ui.overlay.update(frame) }
        } catch (e: Exception) {
            Log.e(TAG, "frame failed", e)
        } finally {
            proxy.close()
        }
    }

    /**
     * Map a box from analysis-image pixels to overlay-view pixels.
     *
     * The preview is letterboxed with fillCenter, so the image is scaled by
     * the LARGER factor and the overflow is cropped equally on both sides.
     * Reproducing that here keeps the drawn boxes on the eyes.
     */
    private fun mapBox(b: IntArray, iw: Int, ih: Int): RectF {
        val vw = ui.overlay.width.toFloat()
        val vh = ui.overlay.height.toFloat()
        if (vw <= 0f || vh <= 0f) return RectF()
        val scale = max(vw / iw, vh / ih)
        val dx = (vw - iw * scale) / 2f
        val dy = (vh - ih * scale) / 2f

        var left = b[0] * scale + dx
        var right = b[2] * scale + dx

        // PreviewView MIRRORS the front camera for display, because a selfie
        // view that moves the wrong way feels broken. ImageAnalysis frames are
        // NOT mirrored. Without correcting for that here, every box lands on
        // the opposite side of the face from the eye it belongs to.
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            val l = vw - right
            right = vw - left
            left = l
        }
        return RectF(left, b[1] * scale + dy, right, b[3] * scale + dy)
    }

    /**
     * RGBA frame -> upright Mat, with exactly one copy.
     *
     * The sensor is mounted rotated relative to the screen, so frames arrive
     * rotated by rotationDegrees. MediaPipe needs an UPRIGHT face - a sideways
     * one is simply not detected - so this must happen before anything else.
     *
     * The plane's ByteBuffer is WRAPPED as a Mat, not copied into one. OpenCV
     * takes the row stride as the `step` argument, so a stride wider than the
     * image - which is common - is handled by the Mat header itself instead
     * of by us reshaping bytes. Core.rotate then makes the single copy that
     * is genuinely needed, into a Mat we already own.
     *
     * The previous version went ByteBuffer -> Bitmap -> Mat -> crop -> rotate:
     * two extra full-frame copies per frame, and a latent crash. Android does
     * not guarantee that the LAST row of a plane carries its padding, so the
     * buffer can be shorter than stride*height, and Bitmap.copyPixelsFromBuffer
     * throws on exactly that. Every frame then failed inside the catch block
     * and the app showed nothing, with only a logcat line to say why.
     */
    private fun ImageProxy.intoUprightMat(dst: Mat): Boolean {
        val plane = planes.firstOrNull() ?: return false
        if (plane.pixelStride != 4) return false     // not the RGBA we asked for
        val buf = plane.buffer
        buf.rewind()
        val src = Mat(height, width, CvType.CV_8UC4, buf, plane.rowStride.toLong())
        when (imageInfo.rotationDegrees) {
            90 -> Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(src, dst, Core.ROTATE_180)
            270 -> Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> src.copyTo(dst)
        }
        // Drops the header only; the pixels belong to the ImageProxy, which
        // onFrame closes once the frame is finished with.
        src.release()
        return !dst.empty()
    }

    override fun onDestroy() {
        super.onDestroy()
        // onCreate bails out before any of this exists if OpenCV fails to load.
        if (!::analysisExecutor.isInitialized) return
        // ORDER MATTERS. shutdown() only refuses NEW work; a frame already
        // executing keeps running. Closing the landmarker and the ONNX session
        // underneath that in-flight frame is a use-after-free in native code -
        // a hard crash on the way out of the app, which users see as "it
        // crashed when I closed it". Cancel, then WAIT for the frame to finish,
        // and only then tear the native resources down.
        analysisExecutor.shutdownNow()
        try {
            analysisExecutor.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS)
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        landmarker?.close(); landmarker = null
        classifier?.close(); classifier = null
        alarm?.release(); alarm = null
        uprightMat.release(); grayMat.release()
        lighting.release()
    }

    companion object {
        private const val TAG = "Drowsy"
        private const val PREF_EAR = "ear_thresh"
    }
}

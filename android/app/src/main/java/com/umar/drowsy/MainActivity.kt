package com.umar.drowsy

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.umar.drowsy.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
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

    // Rolling FPS. A single-frame estimate is far too noisy to read.
    private val frameTimes = ArrayDeque<Double>()
    private var lastFrameT = 0.0

    private val lighting = LightingNormalizer()

    // Allocated once and reused. Creating a Bitmap or Mat inside a 30 fps loop
    // churns megabytes per second and hands the garbage collector work inside
    // the frame budget, which is what makes camera apps stutter.
    private val rawMat = Mat()
    private val uprightMat = Mat()
    private val grayMat = Mat()
    private var rawBitmap: Bitmap? = null
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

        // initLocal() loads the bundled native libs. Without this every OpenCV
        // call throws UnsatisfiedLinkError, so fail loudly rather than crash
        // later in the frame loop.
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show()
            Log.e(TAG, "OpenCVLoader.initLocal() returned false")
        }

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
            Toast.makeText(this, "Keep your eyes OPEN", Toast.LENGTH_SHORT).show()
        }
        ui.btnSwitch.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            startCamera()
        }
        ui.btnAlarm.setOnClickListener {
            val on = alarm?.toggle() ?: false
            ui.btnAlarm.text = if (on) "Mute" else "Unmute"
        }

        // No stored calibration yet, so measure this driver straight away
        // rather than relying on them remembering to press a button.
        calibrator.start()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
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
                // The same principle as the desktop VideoSource: process the
                // NEWEST frame and drop anything that queued up behind it.
                // For a real-time safety signal, fresh beats complete.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // RGBA out of the box saves writing a YUV_420_888 converter.
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { proxy -> onFrame(proxy) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.Builder().requireLensFacing(lensFacing).build(),
                    preview, analysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
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
                val pose = Geometry.headPose(pts, w, h)
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

            val eL = useL; val eR = useR
            val bx = boxes.toList()
            ui.overlay.post {
                ui.overlay.update(st, ear, mar, pitch, yaw, earThresh,
                    wMax, wMin, eL, eR, fps, bx,
                    calibrator.armed, calibrator.remaining(now))
            }
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
     * RGBA frame -> upright Mat, reusing buffers.
     *
     * The sensor is mounted rotated relative to the screen, so frames arrive
     * rotated by rotationDegrees. MediaPipe needs an UPRIGHT face - a sideways
     * one is simply not detected - so this must happen before anything else.
     *
     * Rotation is done with OpenCV on the Mat rather than with a Bitmap
     * Matrix. Bitmap.createBitmap allocates a fresh bitmap every call, and at
     * 30 fps on a 720p frame that is tens of megabytes a second of garbage.
     * Core.rotate writes into a Mat we already own.
     *
     * The row stride is handled explicitly: it is frequently WIDER than the
     * image, and copying it as if it were not skews every row into a diagonal
     * smear.
     */
    private fun ImageProxy.intoUprightMat(dst: Mat): Boolean {
        val plane = planes.firstOrNull() ?: return false
        val padded = width +
            (plane.rowStride - plane.pixelStride * width) / plane.pixelStride

        var bmp = rawBitmap
        if (bmp == null || bmp.width != padded || bmp.height != height) {
            bmp = Bitmap.createBitmap(padded, height, Bitmap.Config.ARGB_8888)
            rawBitmap = bmp
        }
        plane.buffer.rewind()
        bmp.copyPixelsFromBuffer(plane.buffer)
        Utils.bitmapToMat(bmp, rawMat)

        val src = if (padded != width) Mat(rawMat, Rect(0, 0, width, height)) else rawMat
        when (imageInfo.rotationDegrees) {
            90 -> Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(src, dst, Core.ROTATE_180)
            270 -> Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> src.copyTo(dst)
        }
        if (src !== rawMat) src.release()
        return !dst.empty()
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        landmarker?.close()
        classifier?.close()
        alarm?.release()
        rawMat.release(); uprightMat.release(); grayMat.release()
        lighting.release()
    }

    companion object { private const val TAG = "Drowsy" }
}

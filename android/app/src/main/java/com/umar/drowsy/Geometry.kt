package com.umar.drowsy

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Kotlin port of src/geometry.py. Pure maths on facial landmarks, no ML.
 *
 * The landmark INDICES are identical to the desktop version because they are
 * fixed by MediaPipe itself: point 33 is the outer corner of the right eye in
 * every face, on every platform. That is what makes porting this safe.
 */
object Geometry {

    // 6 points per eye: [outer, upper1, upper2, inner, lower2, lower1] so that
    // (p2,p6) and (p3,p5) are vertical pairs and (p1,p4) is horizontal.
    val RIGHT_EYE_EAR = intArrayOf(33, 160, 158, 133, 153, 144)
    val LEFT_EYE_EAR = intArrayOf(362, 385, 387, 263, 373, 380)

    val RIGHT_EYE_CONTOUR = intArrayOf(
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    )
    val LEFT_EYE_CONTOUR = intArrayOf(
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    )

    val MOUTH_UPPER = intArrayOf(82, 13, 312)
    val MOUTH_LOWER = intArrayOf(87, 14, 317)
    val MOUTH_CORNERS = intArrayOf(78, 308)

    private val POSE_LANDMARKS = intArrayOf(1, 199, 33, 263, 61, 291)

    /**
     * A generic 3D head in millimetres, origin at the nose tip. Not a
     * measurement of any particular person: only ANGLES are needed, so a rough
     * shape is sufficient.
     */
    private val MODEL_POINTS_3D = listOf(
        Point3(0.0, 0.0, 0.0),          // nose tip
        Point3(0.0, -63.6, -12.5),      // chin
        Point3(-43.3, 32.7, -26.0),     // right eye outer corner
        Point3(43.3, 32.7, -26.0),      // left eye outer corner
        Point3(-28.9, -28.9, -24.1),    // right mouth corner
        Point3(28.9, -28.9, -24.1)      // left mouth corner
    )

    /** Landmarks arrive NORMALISED to 0..1; scale to pixels before any maths. */
    fun toPixels(lms: List<NormalizedLandmark>, w: Int, h: Int): Array<FloatArray> =
        Array(lms.size) { i -> floatArrayOf(lms[i].x() * w, lms[i].y() * h) }

    private fun dist(a: FloatArray, b: FloatArray): Double =
        hypot((a[0] - b[0]).toDouble(), (a[1] - b[1]).toDouble())

    /**
     * EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)  -- eye height over eye width.
     *
     * Dividing by width is what makes it scale-invariant: lean toward the
     * camera and both grow, so the ratio does not move. Without that division
     * "8 pixels tall" would mean open when close and shut when far.
     */
    fun eyeAspectRatio(pts: Array<FloatArray>, idx: IntArray): Double {
        val p1 = pts[idx[0]]; val p2 = pts[idx[1]]; val p3 = pts[idx[2]]
        val p4 = pts[idx[3]]; val p5 = pts[idx[4]]; val p6 = pts[idx[5]]
        val horizontal = dist(p1, p4)
        if (horizontal < 1e-6) return 0.0
        return (dist(p2, p6) + dist(p3, p5)) / (2.0 * horizontal)
    }

    /**
     * Same idea for the mouth, averaged over three vertical pairs.
     * Talking gives brief spikes; a yawn is high AND sustained, which is why
     * the detector also requires a minimum duration.
     */
    fun mouthAspectRatio(pts: Array<FloatArray>): Double {
        val width = dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
        if (width < 1e-6) return 0.0
        var v = 0.0
        for (i in MOUTH_UPPER.indices) v += dist(pts[MOUTH_UPPER[i]], pts[MOUTH_LOWER[i]])
        return (v / MOUTH_UPPER.size) / width
    }

    /** Corner-to-corner eye width in pixels: our measure of eye visibility. */
    fun eyeWidth(pts: Array<FloatArray>, idx: IntArray): Double =
        dist(pts[idx[0]], pts[idx[3]])

    // Everything solvePnP needs, allocated ONCE. The previous version created
    // and released nine Mats per frame - the 3D model, the camera matrix and
    // the distortion vector among them, none of which ever change.
    private val modelPts = MatOfPoint3f(*MODEL_POINTS_3D.toTypedArray())
    private val imagePts = MatOfPoint2f()
    private val camMat = Mat(3, 3, CvType.CV_64FC1)
    private val distCoeffs = MatOfDouble(0.0, 0.0, 0.0, 0.0)
    private val rvec = Mat()
    private val tvec = Mat()
    private val rot = Mat()
    private val mtxR = Mat()
    private val mtxQ = Mat()

    // Last frame's solution, used to seed the next solve. One driver, so
    // one slot - the same design as _POSE_PREV in geometry.py.
    private val prevRvec = Mat()
    private val prevTvec = Mat()
    private var haveSeed = false

    /** Forget the previous pose, e.g. after the face has been lost. */
    fun resetPoseSeed() { haveSeed = false }

    /**
     * Head orientation via PnP. We know 6 points in 3D and where they landed
     * in 2D; solvePnP recovers the rotation that explains that projection.
     *
     * The iterative solver is SEEDED with the previous frame's answer, as the
     * desktop version is. The head moves only a little between consecutive
     * frames, so last frame's pose is an excellent starting point: the solver
     * converges in fewer iterations and, more importantly, cannot flip to the
     * mirror-image solution that is also a valid local minimum. Unseeded,
     * pitch and yaw occasionally jumped by tens of degrees for a single frame
     * - enough to fake the start of a head nod.
     *
     * Returns (pitch, yaw, roll) in degrees.
     *   pitch < 0 -> tipping down (nodding off)
     *   yaw   != 0 -> looking left/right
     */
    fun headPose(pts: Array<FloatArray>, w: Int, h: Int): DoubleArray {
        imagePts.fromArray(*Array(POSE_LANDMARKS.size) {
            val p = pts[POSE_LANDMARKS[it]]
            Point(p[0].toDouble(), p[1].toDouble())
        })

        // Approximate intrinsics: a camera's focal length in pixels is roughly
        // the image width. Good enough because only angles are wanted.
        val focal = w.toDouble()
        camMat.put(0, 0, focal, 0.0, w / 2.0, 0.0, focal, h / 2.0, 0.0, 0.0, 1.0)

        if (haveSeed) { prevRvec.copyTo(rvec); prevTvec.copyTo(tvec) }
        val ok = try {
            Calib3d.solvePnP(modelPts, imagePts, camMat, distCoeffs, rvec, tvec,
                haveSeed, Calib3d.SOLVEPNP_ITERATIVE)
        } catch (e: Exception) {
            false
        }
        if (!ok) {
            haveSeed = false
            return doubleArrayOf(0.0, 0.0, 0.0)
        }
        rvec.copyTo(prevRvec); tvec.copyTo(prevTvec); haveSeed = true

        Calib3d.Rodrigues(rvec, rot)
        // OpenCV's Java binding returns the Euler angles as a plain double[]
        // (degrees, x/y/z) -- not a Scalar. mtxR and mtxQ are required output
        // arguments even though we only want the angles.
        val angles = Calib3d.RQDecomp3x3(rot, mtxR, mtxQ)
        return doubleArrayOf(wrap(angles[0]), wrap(angles[1]), wrap(angles[2]))
    }

    /**
     * RQDecomp3x3 can report values near +/-180 for a forward-facing head.
     * Wrap them back into a human-readable [-90, 90].
     */
    private fun wrap(a: Double): Double = when {
        a > 90 -> a - 180
        a < -90 -> a + 180
        else -> a
    }

    /**
     * Square crop box around a set of eyelid landmarks.
     *
     * SQUARE matters: the CNN was trained on square 32x32 inputs. A wide
     * rectangle squashed into a square would distort the eye at inference but
     * not during training -- the classic train/serve mismatch.
     */
    fun squareBox(pts: Array<FloatArray>, idx: IntArray, margin: Double): IntArray {
        var x0 = Float.MAX_VALUE; var y0 = Float.MAX_VALUE
        var x1 = -Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
        for (i in idx) {
            x0 = min(x0, pts[i][0]); y0 = min(y0, pts[i][1])
            x1 = max(x1, pts[i][0]); y1 = max(y1, pts[i][1])
        }
        val cx = (x0 + x1) / 2.0; val cy = (y0 + y1) / 2.0
        val half = max(x1 - x0, y1 - y0) * margin / 2.0
        return intArrayOf(
            Math.round(cx - half).toInt(), Math.round(cy - half).toInt(),
            Math.round(cx + half).toInt(), Math.round(cy + half).toInt()
        )
    }
}

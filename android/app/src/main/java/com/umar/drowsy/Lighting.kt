package com.umar.drowsy

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Kotlin port of src/lighting.py. Adaptive night and sun correction.
 *
 * Night and sunlight are the same bug seen from two ends: the pixel values are
 * bunched in the wrong part of the range. Night squashes everything into 0-40
 * so the contrast across an eyelid is a couple of levels. Sun clips at 255,
 * which is destructive - a value never recorded cannot be recovered. Backlit
 * is the nasty one and the common one in a car: bright sky through the
 * windscreen with the face in shadow, so the frame AVERAGE looks perfectly
 * normal while the face itself is crushed.
 *
 * So this does not detect "night" or "day". It measures where the histogram
 * sits and moves it to the middle - one mechanism, every condition.
 *
 * Measured on the desktop version with simulated conditions:
 *     night       P(closed) 0.557 -> 0.169   (was calling open eyes shut)
 *     deep night  P(closed) 0.785 -> 0.274
 *     conditions handled correctly 5/7 -> 7/7
 *
 * This runs BEFORE MediaPipe, on the whole frame. Enhancing only the eye crop
 * would be too late: in the dark there is no face to find, so there is no crop.
 */

enum class Light(val text: String) {
    DARK("NIGHT"),
    DIM("DIM"),
    NORMAL("NORMAL"),
    BRIGHT("BRIGHT SUN"),
    BACKLIT("BACKLIT")
}

data class LightStats(
    val condition: Light = Light.NORMAL,
    val mean: Double = 0.0,
    val faceMean: Double = 0.0,
    val gamma: Double = 1.0
)

class LightingNormalizer(var enabled: Boolean = true) {

    // A frame anywhere in this band is already well exposed and is LEFT ALONE.
    //
    // Without the dead band, gamma forced every frame to a single target, so a
    // perfectly good bright scene at mean 183 was darkened with gamma 2.19 -
    // the clamp ceiling - for no benefit, and pushed away from the statistics
    // the model was trained on.
    //
    // Correcting only toward the nearest EDGE also keeps gamma continuous: at
    // mean == COMFORT_LOW the correction is exactly 1.0, so a frame hovering
    // at the boundary cannot flicker between corrected and uncorrected.
    private val comfortLow = 85.0
    private val comfortHigh = 165.0
    private val smooth = 0.12

    private var gammaNow = 1.0
    var stats = LightStats(); private set

    private val claheStrong: CLAHE = Imgproc.createCLAHE(3.0, Size(8.0, 8.0))
    private val claheSoft: CLAHE = Imgproc.createCLAHE(1.6, Size(8.0, 8.0))

    // Every Mat is allocated once. Allocating inside a 30 fps loop is what
    // makes Android apps stutter: each frame would otherwise churn several
    // megabytes and hand the garbage collector work during the frame budget.
    private val bgr = Mat()
    private val lab = Mat()
    private val l = Mat()
    private val small = Mat()
    private val lut = Mat(1, 256, CvType.CV_8U)
    private val lutData = ByteArray(256)
    private val smallBuf = ByteArray(160 * 90)
    private val hist = IntArray(256)

    /**
     * The gamma that maps `mean` onto the target.
     *
     * We want (mean/255)^gamma == target/255, so taking logs and rearranging
     * gives the expression below. Clamped because an unclamped gamma on a
     * nearly black frame goes enormous and amplifies pure sensor noise into a
     * grey blizzard.
     */
    private fun autoGamma(mean: Double): Double {
        val m = mean.coerceIn(4.0, 250.0)
        if (m in comfortLow..comfortHigh) return 1.0
        val target = if (m < comfortLow) comfortLow else comfortHigh
        val g = ln(target / 255.0) / ln(m / 255.0)
        return g.coerceIn(0.35, 2.2)
    }

    private fun buildLut(gamma: Double) {
        for (i in 0..255) {
            val v = ((i / 255.0).pow(gamma) * 255.0).coerceIn(0.0, 255.0)
            // Java bytes are signed; values above 127 must wrap deliberately.
            lutData[i] = v.toInt().toByte()
        }
        lut.put(0, 0, lutData)
    }

    /**
     * @param rgba an RGBA frame, modified IN PLACE.
     * @return the same Mat, enhanced.
     */
    fun process(rgba: Mat): Mat {
        if (rgba.empty()) return rgba

        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        // LAB, not BGR: applying gamma to B, G and R separately changes their
        // ratios and shifts skin tone, which hurts face detection. L is
        // lightness alone, so brightness moves and hue does not.
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)

        // Pull out L only. Core.split would hand back three freshly
        // allocated Mats every frame - the A and B planes among them, which
        // are never touched - and that contradicted the "allocated once"
        // promise above: ~2.7 MB of churn per 720p frame for nothing.
        Core.extractChannel(lab, l, 0)

        // Histogram summary on a downscaled copy: we need the distribution's
        // shape, not precision, and doing this at full resolution every frame
        // is wasted work.
        Imgproc.resize(l, small, Size(160.0, 90.0), 0.0, 0.0, Imgproc.INTER_AREA)

        // One histogram pass yields mean, the 5th/95th percentiles and the
        // clipped fraction together. Cheaper than three separate OpenCV calls
        // over the same 14,400 pixels, and it is the only way to get
        // percentiles, which OpenCV does not expose directly.
        small.get(0, 0, smallBuf)
        java.util.Arrays.fill(hist, 0)
        for (b in smallBuf) hist[b.toInt() and 0xFF]++
        val total = smallBuf.size

        var sum = 0L
        for (i in 0..255) sum += i.toLong() * hist[i]
        val mean = sum.toDouble() / total

        var cum = 0
        var p5 = 0
        var p95 = 255
        val loTarget = (total * 0.05).toInt()
        val hiTarget = (total * 0.95).toInt()
        for (i in 0..255) { cum += hist[i]; if (cum >= loTarget) { p5 = i; break } }
        cum = 0
        for (i in 0..255) { cum += hist[i]; if (cum >= hiTarget) { p95 = i; break } }
        // How much of the available range the image actually uses.
        val contrast = (p95 - p5).toDouble()

        var hiCount = 0
        for (i in 250..255) hiCount += hist[i]
        val hiFrac = hiCount.toDouble() / total

        // Centre crop approximates "how bright is the face" without running a
        // detector - in a driver-facing camera the face is near the middle.
        val face = Mat(small, Rect(40, 18, 80, 54))
        val faceMean = Core.mean(face).`val`[0]
        face.release()

        val cond = classify(mean, faceMean, contrast, hiFrac)

        if (!enabled) {
            stats = LightStats(cond, mean, faceMean, 1.0)
            return rgba
        }

        // For a backlit frame the FACE is what must be exposed correctly, not
        // the average of a scene dominated by bright sky. Aim at the centre
        // and let the background clip - the background is not what we need.
        val aim = if (cond == Light.BACKLIT) faceMean else mean
        val target = autoGamma(aim)

        // Ease toward the target rather than snapping. Recomputing gamma from
        // scratch every frame would make passing headlights pulse the whole
        // image, jittering the landmarks and faking blinks through EAR wobble.
        gammaNow += (target - gammaNow) * smooth

        if (kotlin.math.abs(gammaNow - 1.0) > 0.03) {
            buildLut(gammaNow)
            Core.LUT(l, lut, l)
        }

        // Strong local contrast only where the image is genuinely poor. The
        // strong setting on an already-good frame just adds noise and halos
        // around the eyelid we are trying to measure.
        if (cond == Light.NORMAL) claheSoft.apply(l, l) else claheStrong.apply(l, l)

        if (cond == Light.DARK) {
            // Brightening a night frame brightens its sensor noise too. A 3x3
            // median kills speckle while preserving edges, unlike a blur which
            // would soften the eyelid.
            Imgproc.medianBlur(l, l, 3)
        }

        Core.insertChannel(l, lab, 0)
        Imgproc.cvtColor(lab, bgr, Imgproc.COLOR_Lab2BGR)
        Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA)

        stats = LightStats(cond, mean, faceMean, gammaNow)
        return rgba
    }

    /**
     * Order matters. BACKLIT is checked FIRST because it defeats every simpler
     * test: the frame mean can look entirely normal while the face sits in
     * deep shadow. Only comparing the centre against the whole frame catches it.
     */
    private fun classify(
        mean: Double, faceMean: Double, contrast: Double, hiFrac: Double
    ): Light = when {
        hiFrac > 0.12 && faceMean < mean - 25 -> Light.BACKLIT
        // The contrast term matters: without it a normally-lit frame whose
        // CENTRE happens to be dark - dark clothing, a beard, a shadow - is
        // misread as night and gets median-blurred for no reason.
        mean < 45 || (faceMean < 40 && contrast < 60) -> Light.DARK
        mean < 80 -> Light.DIM
        mean > 185 || hiFrac > 0.28 -> Light.BRIGHT
        else -> Light.NORMAL
    }

    /**
     * Forget the eased gamma. Call when the scene changes discontinuously -
     * a camera switch, for instance - so the next frame is not corrected
     * with a value tuned for a different picture.
     */
    fun reset() { gammaNow = 1.0 }

    fun toggle(): Boolean { enabled = !enabled; return enabled }

    fun release() {
        bgr.release(); lab.release(); l.release(); small.release(); lut.release()
    }
}

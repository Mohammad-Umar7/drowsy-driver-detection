package com.umar.drowsy

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/**
 * Runs the exported eyenet.onnx on device.
 *
 * =========================================================================
 * THE ONE THING THAT MATTERS IN THIS FILE
 * =========================================================================
 *
 * Preprocessing here must be BYTE-FOR-BYTE what src/preprocess.py did during
 * training. If it differs even slightly, the model still returns confident
 * numbers -- they are just wrong, and nothing in the app can tell. That
 * failure is called train/serve skew and it is silent by nature.
 *
 * preprocess.py does exactly four things:
 *     1. to grayscale
 *     2. resize to 32x32 with INTER_AREA
 *     3. CLAHE, clipLimit 2.0, tiles 4x4
 *     4. scale to [0,1]
 *
 * Each is reproduced below with the SAME OpenCV implementation, which is
 * precisely why OpenCV is a dependency of this app rather than a hand-written
 * CLAHE in Kotlin. Re-implementing it would mean matching histogram binning,
 * clip redistribution and bilinear tile interpolation exactly, and any
 * mismatch would be invisible.
 *
 * INTER_AREA is not interchangeable with INTER_LINEAR either: for downscaling
 * INTER_AREA averages the source pixels, while INTER_LINEAR samples and
 * aliases. Training used INTER_AREA.
 */
class EyeClassifier(context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val clahe: CLAHE = Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
    private val size = 32

    // Reused across frames so we are not allocating 30 times a second.
    private val buffer: FloatBuffer = FloatBuffer.allocate(2 * size * size)
    private val gray32 = Mat()
    private val work = Mat()

    init {
        val bytes = context.assets.open("eyenet.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            // Two 32x32 images is a tiny workload. More threads would spend
            // more time coordinating than computing, and would compete with
            // MediaPipe, which is the actual bottleneck.
            setIntraOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
    }

    /**
     * Crop with zero-free padding when the box runs off the frame edge.
     *
     * Naive cropping silently returns a SMALLER image for a partially
     * off-screen box, which then gets resized and warps the eye. Replicating
     * the border keeps the geometry honest.
     */
    private fun safeCrop(grayFrame: Mat, box: IntArray): Mat {
        val w = grayFrame.cols(); val h = grayFrame.rows()
        var x0 = box[0]; var y0 = box[1]; var x1 = box[2]; var y1 = box[3]
        val padL = maxOf(0, -x0); val padT = maxOf(0, -y0)
        val padR = maxOf(0, x1 - w); val padB = maxOf(0, y1 - h)
        x0 = x0.coerceIn(0, w); y0 = y0.coerceIn(0, h)
        x1 = x1.coerceIn(0, w); y1 = y1.coerceIn(0, h)
        if (x1 <= x0 || y1 <= y0) return Mat.zeros(1, 1, CvType.CV_8UC1)

        val roi = Mat(grayFrame, Rect(x0, y0, x1 - x0, y1 - y0))
        if (padL or padT or padR or padB == 0) return roi
        val padded = Mat()
        Core.copyMakeBorder(roi, padded, padT, padB, padL, padR, Core.BORDER_REPLICATE)
        roi.release()
        return padded
    }

    /** One eye crop -> 32x32 floats in [0,1], written into `dst` at `offset`. */
    private fun preprocessInto(grayFrame: Mat, box: IntArray, dst: FloatArray, offset: Int) {
        val crop = safeCrop(grayFrame, box)
        // 2. INTER_AREA, matching preprocess.py
        Imgproc.resize(crop, work, Size(size.toDouble(), size.toDouble()), 0.0, 0.0,
            Imgproc.INTER_AREA)
        // 3. the same CLAHE settings used in training
        clahe.apply(work, work)
        // 4. to [0,1]
        work.convertTo(gray32, CvType.CV_32F, 1.0 / 255.0)
        val tmp = FloatArray(size * size)
        gray32.get(0, 0, tmp)
        System.arraycopy(tmp, 0, dst, offset, tmp.size)
        crop.release()
    }

    /**
     * @return P(closed) for [left, right]. The exported graph already applies
     *         softmax, so these are probabilities, not logits.
     */
    fun classify(grayFrame: Mat, boxLeft: IntArray, boxRight: IntArray): DoubleArray {
        val data = FloatArray(2 * size * size)
        preprocessInto(grayFrame, boxLeft, data, 0)
        preprocessInto(grayFrame, boxRight, data, size * size)

        buffer.rewind()
        buffer.put(data)
        buffer.rewind()

        // Both eyes go through as ONE batch of 2. Two separate calls would pay
        // the session-run overhead twice, and on a model this small that
        // overhead dwarfs the actual arithmetic.
        val shape = longArrayOf(2, 1, size.toLong(), size.toLong())
        OnnxTensor.createTensor(env, buffer, shape).use { tensor ->
            session.run(mapOf("eye" to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out = result[0].value as Array<FloatArray>
                // column 1 = P(closed); column 0 = P(open)
                return doubleArrayOf(out[0][1].toDouble(), out[1][1].toDouble())
            }
        }
    }

    fun close() {
        session.close()
        gray32.release(); work.release()
    }
}

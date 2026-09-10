package com.umar.drowsy

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Transparent HUD drawn above the camera preview.
 *
 * It is a separate View rather than annotations burned into the camera frame:
 * drawing onto the frame would mean copying every frame just to decorate it,
 * and the copy would cost more than the whole detector.
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private var state: DrowsyState = DrowsyState()
    private var ear = 0.0
    private var mar = 0.0
    private var pitch = 0.0
    private var yaw = 0.0
    private var earThresh = Cfg.EAR_THRESH
    private var widthMax = 0.0
    private var widthMin = 0.0
    private var useL = true
    private var useR = true
    private var fps = 0.0
    private var calibrating = false
    private var calibRemaining = 0.0
    private var eyeBoxes: List<RectF> = emptyList()

    private val big = Paint().apply {
        isAntiAlias = true; textSize = 62f; isFakeBoldText = true
    }
    private val text = Paint().apply {
        isAntiAlias = true; textSize = 30f; color = Color.rgb(225, 232, 238)
    }
    private val small = Paint().apply {
        isAntiAlias = true; textSize = 26f; color = Color.rgb(170, 184, 195)
    }
    private val panel = Paint().apply { color = Color.argb(150, 0, 0, 0) }
    private val barBg = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 2f; color = Color.rgb(90, 100, 110)
    }
    private val barFill = Paint()
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 3f; isAntiAlias = true
    }
    private val border = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 26f; color = Color.rgb(220, 0, 0)
    }

    fun update(
        s: DrowsyState, ear: Double, mar: Double, pitch: Double, yaw: Double,
        earThresh: Double, wMax: Double, wMin: Double, useL: Boolean, useR: Boolean,
        fps: Double, boxes: List<RectF>, calibrating: Boolean, calibRemaining: Double
    ) {
        this.state = s; this.ear = ear; this.mar = mar
        this.pitch = pitch; this.yaw = yaw; this.earThresh = earThresh
        this.widthMax = wMax; this.widthMin = wMin
        this.useL = useL; this.useR = useR; this.fps = fps
        this.eyeBoxes = boxes
        this.calibrating = calibrating; this.calibRemaining = calibRemaining
        postInvalidateOnAnimation()
    }

    private fun levelColor(l: Level): Int = when (l) {
        Level.NO_FACE -> Color.rgb(150, 150, 150)
        Level.AWAKE -> Color.rgb(90, 210, 110)
        Level.DISTRACTED -> Color.rgb(255, 190, 40)
        Level.DROWSY -> Color.rgb(255, 140, 30)
        Level.CRITICAL -> Color.rgb(255, 60, 60)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        val col = levelColor(state.level)

        // Eye boxes, coloured by state, so it is obvious what is being watched.
        boxPaint.color = if (state.closed) Color.rgb(235, 60, 60) else Color.rgb(0, 220, 140)
        for (b in eyeBoxes) canvas.drawRect(b, boxPaint)

        // Flashing border on CRITICAL. Peripheral vision picks up motion even
        // when the driver is not looking directly at the phone.
        if (state.level == Level.CRITICAL &&
            (System.currentTimeMillis() / 180) % 2 == 0L) {
            canvas.drawRect(13f, 13f, w - 13f, h - 13f, border)
        }

        // ---- top banner ----
        canvas.drawRect(0f, 0f, w, 190f, panel)
        big.color = col
        canvas.drawText(state.level.text, 26f, 82f, big)
        small.color = Color.rgb(200, 210, 218)
        canvas.drawText(state.reasons.take(2).joinToString(" | "), 26f, 122f, small)
        canvas.drawText("%.0f FPS".format(fps), w - 130f, 122f, small)

        if (calibrating) {
            text.color = Color.rgb(255, 220, 60)
            canvas.drawText("CALIBRATING - keep eyes OPEN  %.1fs".format(calibRemaining),
                26f, 166f, text)
            text.color = Color.rgb(225, 232, 238)
        }

        // ---- bottom metrics ----
        val top = h - 250f
        canvas.drawRect(0f, top, 560f, h, panel)
        var y = top + 44f

        drawBar(canvas, 24f, y - 22f, 260f, 20f, state.perclos,
            if (state.perclos >= Cfg.PERCLOS_WARN) Color.rgb(255, 150, 40)
            else Color.rgb(90, 200, 110),
            "PERCLOS %.1f%%".format(state.perclos * 100), Cfg.PERCLOS_WARN)
        y += 44f

        drawBar(canvas, 24f, y - 22f, 260f, 20f, state.closedScore,
            if (state.closed) Color.rgb(235, 60, 60) else Color.rgb(90, 200, 110),
            "closed %.2f".format(state.closedScore), 0.5)
        y += 44f

        canvas.drawText("EAR %.3f (thr %.3f)  MAR %.2f".format(ear, earThresh, mar),
            24f, y, text)
        y += 38f
        canvas.drawText("pitch %+.0f  yaw %+.0f".format(pitch, yaw), 24f, y, text)
        y += 38f

        // Eye size and which eyes are trusted. This line is what makes the
        // "head turned = false closure" failure visible instead of mysterious.
        val tag = (if (useL) "L" else "-") + (if (useR) "R" else "-")
        text.color = if (state.eyesReliable) Color.rgb(225, 232, 238)
                     else Color.rgb(255, 190, 40)
        canvas.drawText("eye px %.0f/%.0f  using [%s]%s".format(
            widthMax, widthMin, tag,
            if (state.eyesReliable) "" else "  UNRELIABLE"), 24f, y, text)
        text.color = Color.rgb(225, 232, 238)
        y += 38f
        canvas.drawText("blinks ${state.blinks}   yawns ${state.yawns}", 24f, y, text)
    }

    private fun drawBar(
        c: Canvas, x: Float, y: Float, w: Float, h: Float,
        frac: Double, color: Int, label: String, warnAt: Double
    ) {
        c.drawRect(x, y, x + w, y + h, barBg)
        val fill = (w * min(1.0, max(0.0, frac))).toFloat()
        if (fill > 1f) {
            barFill.color = color
            c.drawRect(x + 1f, y + 1f, x + fill - 1f, y + h - 1f, barFill)
        }
        // Tick marking the threshold, so the bar shows how close it is, not
        // just how full it is.
        val wx = x + (w * warnAt).toFloat()
        c.drawLine(wx, y - 4f, wx, y + h + 4f, barBg)
        c.drawText(label, x + w + 14f, y + h - 2f, small)
    }
}

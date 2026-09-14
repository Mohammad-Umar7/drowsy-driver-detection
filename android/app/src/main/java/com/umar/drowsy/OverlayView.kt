package com.umar.drowsy

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.text.TextPaint
import android.text.TextUtils
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Everything the overlay needs to draw one frame, in one object.
 *
 * Built on the analysis thread, handed to the view on the UI thread. An
 * immutable snapshot means the two threads never share mutable state, and a
 * frame that arrives while the previous one is still being drawn simply
 * replaces it.
 */
data class HudFrame(
    val state: DrowsyState = DrowsyState(),
    val faceFound: Boolean = false,
    val ear: Double = 0.0,
    val mar: Double = 0.0,
    val pitch: Double = 0.0,
    val yaw: Double = 0.0,
    val earThresh: Double = Cfg.EAR_THRESH,
    val widthMax: Double = 0.0,
    val widthMin: Double = 0.0,
    val useL: Boolean = false,
    val useR: Boolean = false,
    val fps: Double = 0.0,
    val boxes: List<RectF> = emptyList(),
    val calibrating: Boolean = false,
    val calibRemaining: Double = 0.0,
    val light: LightStats = LightStats(),
    val lightingOn: Boolean = true,
    val alarmOn: Boolean = true,
    val modelOn: Boolean = true,
    val now: Double = 0.0
)

/**
 * Transparent HUD drawn above the camera preview.
 *
 * It is a separate View rather than annotations burned into the camera frame:
 * drawing onto the frame would mean copying every frame just to decorate it,
 * and the copy would cost more than the whole detector.
 *
 * Every size here is in dp or sp, never raw pixels. The first version used
 * pixel literals tuned on one phone, so on a 720p screen the panels covered
 * half the preview and on a 1440p one the text was unreadable.
 *
 * The sparkline is the same idea as the desktop display: the last 30 s of the
 * fused closure score, red where the eyes were shut. A blink is a narrow
 * spike, a microsleep a wide red block, and fatigue is spikes getting wider
 * and closer together - the whole detector in one strip.
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private var f = HudFrame()

    // (time, score, closed) at ~10 Hz over the PERCLOS window.
    private class Sample(val t: Double, val score: Double, val closed: Boolean)
    private val hist = ArrayDeque<Sample>()
    private var lastPush = -1.0
    private val window = Cfg.PERCLOS_WINDOW_SEC

    // ---- palette ----
    private val panelColor = Color.argb(172, 14, 16, 22)
    private val trackColor = Color.rgb(40, 44, 54)
    private val ink = Color.rgb(238, 240, 242)
    private val muted = Color.rgb(150, 160, 172)
    private val dim = Color.rgb(100, 108, 120)
    private val good = Color.rgb(120, 205, 110)
    private val warn = Color.rgb(255, 190, 40)
    private val hot = Color.rgb(255, 140, 30)
    private val bad = Color.rgb(235, 70, 70)
    private val cool = Color.rgb(120, 200, 235)

    private fun levelColor(l: Level): Int = when (l) {
        Level.NO_FACE -> Color.rgb(165, 165, 165)
        Level.AWAKE -> good
        Level.DISTRACTED -> warn
        Level.DROWSY -> hot
        Level.CRITICAL -> bad
    }

    // ---- units ----
    private val dm = resources.displayMetrics
    private fun dp(v: Float) = v * dm.density
    private fun sp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, dm)

    // ---- paints, allocated once ----
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val textPill = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.DEFAULT_BOLD; textSize = sp(19f)
    }
    private val textBody = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { textSize = sp(13.5f) }
    private val textSmall = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { textSize = sp(12f) }
    private val textTiny = TextPaint(Paint.ANTI_ALIAS_FLAG).apply { textSize = sp(10.5f) }
    private val textChip = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.DEFAULT_BOLD; textSize = sp(11.5f)
    }
    private val sparkPath = Path()
    private val rect = RectF()

    /** Called on the UI thread with a snapshot of the frame just analysed. */
    fun update(frame: HudFrame) {
        f = frame
        push(frame.state.closedScore, frame.state.closed, frame.now)
        postInvalidateOnAnimation()
    }

    /** Forget the sparkline history, e.g. when the counters are reset. */
    fun clearHistory() { hist.clear(); lastPush = -1.0 }

    private fun push(score: Double, closed: Boolean, now: Double) {
        if (now <= 0.0) return
        val last = hist.lastOrNull()
        if (lastPush >= 0 && now - lastPush < 0.1 && last != null && last.closed == closed) return
        hist.addLast(Sample(now, score, closed))
        lastPush = now
        val cut = now - window
        while (hist.isNotEmpty() && hist.first().t < cut) hist.removeFirst()
    }

    // ---- primitives ----
    private fun panel(c: Canvas, l: Float, t: Float, r: Float, b: Float, radius: Float,
                      color: Int = panelColor) {
        fill.color = color
        rect.set(l, t, r, b)
        c.drawRoundRect(rect, radius, radius, fill)
    }

    private fun baseline(p: Paint, top: Float, h: Float): Float {
        val fm = p.fontMetrics
        return top + h / 2f - (fm.ascent + fm.descent) / 2f
    }

    /** A filled pill with a label. Returns its width. */
    private fun chip(c: Canvas, s: String, x: Float, y: Float, h: Float, fg: Int, bg: Int,
                     p: TextPaint = textChip, padX: Float = dp(10f)): Float {
        val w = p.measureText(s) + 2 * padX
        panel(c, x, y, x + w, y + h, h / 2f, bg)
        p.color = fg
        c.drawText(s, x + padX, baseline(p, y, h), p)
        return w
    }

    private fun bar(c: Canvas, x: Float, y: Float, w: Float, h: Float, frac: Double,
                    color: Int, warnAt: Double?) {
        panel(c, x, y, x + w, y + h, h / 2f, trackColor)
        val fw = (w * min(1.0, max(0.0, frac))).toFloat()
        if (fw >= h) panel(c, x, y, x + fw, y + h, h / 2f, color)
        else if (fw > 0f) { fill.color = color; c.drawCircle(x + h / 2f, y + h / 2f, h / 2f, fill) }
        if (warnAt != null) {
            val wx = x + (w * warnAt).toFloat()
            stroke.color = ink; stroke.strokeWidth = dp(1.5f)
            c.drawLine(wx, y - dp(3f), wx, y + h + dp(3f), stroke)
        }
    }

    private fun sparkline(c: Canvas, x: Float, y: Float, w: Float, h: Float, now: Double) {
        panel(c, x, y, x + w, y + h, dp(8f), trackColor)
        val cap = dp(15f)
        textTiny.color = dim
        c.drawText("eye closure, last %.0f s".format(window), x + dp(8f), y + dp(11f), textTiny)
        if (hist.size < 2 || now <= 0.0) return
        val py0 = y + cap
        val py1 = y + h - dp(3f)
        stroke.color = Color.rgb(70, 76, 90); stroke.strokeWidth = dp(1f)
        c.drawLine(x + dp(6f), (py0 + py1) / 2f, x + w - dp(6f), (py0 + py1) / 2f, stroke)

        val t0 = now - window
        val xs = FloatArray(hist.size)
        val ys = FloatArray(hist.size)
        var i = 0
        for (s in hist) {
            val px = x + dp(2f) + ((s.t - t0) / window * (w - dp(4f))).toFloat()
            xs[i] = px.coerceIn(x + dp(2f), x + w - dp(2f))
            ys[i] = py1 - (s.score.coerceIn(0.0, 1.0) * (py1 - py0)).toFloat()
            i++
        }
        sparkPath.rewind()
        sparkPath.moveTo(xs[0], py1)
        for (k in xs.indices) sparkPath.lineTo(xs[k], ys[k])
        sparkPath.lineTo(xs[xs.size - 1], py1)
        sparkPath.close()
        fill.color = Color.argb(110, 60, 110, 70)
        c.drawPath(sparkPath, fill)

        stroke.strokeWidth = dp(2f)
        var k = 0
        var prev: Sample? = null
        for (s in hist) {
            if (prev != null) {
                stroke.color = if (prev.closed || s.closed) bad else good
                c.drawLine(xs[k - 1], ys[k - 1], xs[k], ys[k], stroke)
            }
            prev = s; k++
        }
    }

    // ---- the whole thing ----
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val fr = f
        val st = fr.state
        val w = width.toFloat()
        val h = height.toFloat()
        val m = dp(12f)
        val radius = dp(16f)
        val col = levelColor(st.level)

        // Eye boxes, coloured by state, so it is obvious what is being watched.
        stroke.strokeWidth = dp(2f)
        stroke.color = if (st.closed) bad else good
        for (b in fr.boxes) canvas.drawRoundRect(b, dp(6f), dp(6f), stroke)

        // Flashing frame on CRITICAL. Peripheral vision picks up motion even
        // when the driver is not looking directly at the phone.
        val flash = st.level == Level.CRITICAL && (System.currentTimeMillis() / 200) % 2 == 0L
        if (flash) {
            stroke.color = bad; stroke.strokeWidth = dp(12f)
            canvas.drawRect(0f, 0f, w, h, stroke)
        }

        // ---- top card ----
        val topH = dp(92f)
        panel(canvas, m, m, w - m, m + topH, radius)
        var x = m + dp(12f)
        val pillH = dp(36f)
        val pillY = m + dp(12f)
        chip(canvas, st.level.text, x, pillY, pillH,
             if (flash) bad else Color.rgb(14, 16, 22), if (flash) ink else col,
             textPill, dp(14f))

        // Right-hand chips: only the ones carrying information right now.
        var rx = w - m - dp(12f)
        val chipH = dp(24f)
        val chipY = pillY + (pillH - chipH) / 2f
        fun rightChip(s: String, fg: Int, bg: Int) {
            val cw = textChip.measureText(s) + dp(20f)
            rx -= cw
            chip(canvas, s, rx, chipY, chipH, fg, bg)
            rx -= dp(6f)
        }
        rightChip("%.0f FPS".format(fr.fps), muted, trackColor)
        val lightOk = fr.light.condition == Light.NORMAL && fr.lightingOn
        val lightTag = fr.light.condition.text + if (fr.lightingOn) "" else " (fix off)"
        rightChip(lightTag, if (lightOk) muted else Color.rgb(14, 16, 22),
                  if (lightOk) trackColor else warn)
        if (!fr.alarmOn) rightChip("MUTED", ink, dim)
        if (!fr.modelOn) rightChip("EAR only", Color.rgb(14, 16, 22), cool)

        // Reasons, ellipsized to the space that is actually left.
        textSmall.color = muted
        val reason = st.reasons.take(2).joinToString("  |  ")
        val avail = (w - m - dp(12f)) - x
        canvas.drawText(
            TextUtils.ellipsize(reason, textSmall, avail, TextUtils.TruncateAt.END).toString(),
            x, m + topH - dp(14f), textSmall)

        // ---- calibration banner ----
        if (fr.calibrating) {
            val bh = dp(56f)
            val bt = m + topH + dp(10f)
            panel(canvas, m, bt, w - m, bt + bh, dp(12f), Color.argb(215, 14, 16, 22))
            textBody.color = warn
            val msg = if (fr.faceFound) "CALIBRATING  -  keep your eyes OPEN"
                      else "CALIBRATING  -  waiting for a face..."
            canvas.drawText(msg, m + dp(14f), bt + dp(24f), textBody)
            val frac = if (fr.faceFound) 1.0 - fr.calibRemaining / 3.0 else 0.0
            bar(canvas, m + dp(14f), bt + dp(36f), w - 2 * m - dp(28f), dp(8f), frac, warn, null)
        }

        // ---- bottom card, above the button row ----
        val btnInset = dp(74f)
        val cardH = dp(214f)
        val top = h - btnInset - cardH
        panel(canvas, m, top, w - m, h - btnInset, radius)
        val ix = m + dp(14f)
        val iw = w - 2 * m - dp(28f)
        var y = top + dp(16f)
        val labW = dp(66f)
        val valW = dp(88f)
        val bw = iw - labW - valW
        val rowH = dp(14f)

        textSmall.color = muted
        canvas.drawText("PERCLOS", ix, baseline(textSmall, y, rowH), textSmall)
        val pc = if (st.perclos >= Cfg.PERCLOS_CRITICAL) bad
                 else if (st.perclos >= Cfg.PERCLOS_WARN) hot else good
        bar(canvas, ix + labW, y, bw, rowH, st.perclos, pc, Cfg.PERCLOS_WARN)
        textBody.color = ink
        canvas.drawText("%.1f%%".format(st.perclos * 100), ix + labW + bw + dp(10f),
                        baseline(textBody, y, rowH), textBody)
        y += dp(26f)

        textSmall.color = muted
        canvas.drawText("EYES", ix, baseline(textSmall, y, rowH), textSmall)
        bar(canvas, ix + labW, y, bw, rowH, st.closedScore, if (st.closed) bad else good, 0.5)
        textBody.color = if (st.closed) bad else ink
        canvas.drawText(if (st.closed) "CLOSED %.1fs".format(st.closureSec) else "open",
                        ix + labW + bw + dp(10f), baseline(textBody, y, rowH), textBody)
        y += dp(24f)

        sparkline(canvas, ix, y, iw, dp(48f), fr.now)
        y += dp(48f) + dp(22f)

        if (fr.faceFound) {
            textBody.color = ink
            canvas.drawText("EAR %.3f".format(fr.ear), ix, y, textBody)
            textSmall.color = dim
            canvas.drawText("thr %.3f".format(fr.earThresh), ix + dp(78f), y, textSmall)
            textBody.color = ink
            canvas.drawText("MAR %.2f".format(fr.mar), ix + dp(148f), y, textBody)
            val tag = (if (fr.useL) "L" else "-") + (if (fr.useR) "R" else "-")
            textSmall.color = if (st.eyesReliable) ink else warn
            val eyes = "eyes [%s] %.0fpx".format(tag, fr.widthMax)
            canvas.drawText(eyes, ix + iw - textSmall.measureText(eyes), y, textSmall)
            y += dp(22f)
            textSmall.color = muted
            canvas.drawText("pitch %+.0f   yaw %+.0f".format(fr.pitch, fr.yaw), ix, y, textSmall)
            y += dp(22f)
        } else {
            textBody.color = dim
            canvas.drawText("no face in view", ix, y, textBody)
            y += dp(44f)
        }
        textSmall.color = muted
        canvas.drawText("blinks ${st.blinks}    long ${st.longBlinks} " +
                        "(%.0f/min)    yawns ${st.yawns}".format(st.longBlinkRate),
                        ix, y, textSmall)
    }
}

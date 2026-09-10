package com.umar.drowsy

import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.HandlerThread
import android.os.Vibrator
import android.os.VibrationEffect
import android.content.Context
import android.os.Build

/**
 * Escalating audible + haptic alarm.
 *
 * THE RULE: never block the camera thread. Playing a one-second tone inline
 * would stall frame processing for that whole second, exactly when the driver
 * most needs the system working. Every sound is therefore queued onto a
 * dedicated background thread and the caller returns immediately.
 *
 * ToneGenerator is used rather than bundled audio files because it is
 * synthesised by the OS: no assets, no decoding, no latency, and it plays on
 * STREAM_ALARM so it is audible even when the phone is on silent -- which
 * matters, since a phone in a car mount is usually silenced.
 */
class Alarm(context: Context) {

    var enabled = true

    private val thread = HandlerThread("alarm").apply { start() }
    private val handler = Handler(thread.looper)
    private val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator

    // STREAM_ALARM, at full volume. A drowsiness warning that the ringer
    // switch can silence is not a safety feature.
    private val tone = try {
        ToneGenerator(AudioManager.STREAM_ALARM, 100)
    } catch (e: RuntimeException) {
        null
    }

    @Volatile private var playing = false

    /**
     * Three deliberately different sounds. Distraction is the quietest and
     * shortest: it is a reminder, not an emergency, and an aggressive tone for
     * a mirror check would train the driver to ignore every alert.
     */
    private fun patternFor(kind: String): List<Pair<Int, Int>> = when (kind) {
        "distract" -> listOf(ToneGenerator.TONE_PROP_BEEP to 120)
        "critical" -> listOf(
            ToneGenerator.TONE_CDMA_HIGH_L to 250,
            ToneGenerator.TONE_CDMA_HIGH_L to 250,
            ToneGenerator.TONE_CDMA_HIGH_L to 400
        )
        else -> listOf(
            ToneGenerator.TONE_PROP_BEEP to 200,
            ToneGenerator.TONE_PROP_BEEP to 200
        )
    }

    fun fire(kind: String) {
        if (!enabled || playing) return
        playing = true
        handler.post {
            try {
                for ((t, ms) in patternFor(kind)) {
                    tone?.startTone(t, ms)
                    Thread.sleep(ms + 60L)
                }
                // Vibration matters in a car: road noise can bury a beep, but
                // a phone buzzing in a mount is still felt and seen.
                if (kind == "critical") vibrate(longArrayOf(0, 400, 150, 400))
                else if (kind == "drowsy") vibrate(longArrayOf(0, 250))
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            } catch (e: Exception) {
                // Audio must never take the detector down with it.
            } finally {
                playing = false
            }
        }
    }

    private fun vibrate(pattern: LongArray) {
        val v = vibrator ?: return
        if (!v.hasVibrator()) return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            v.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(pattern, -1)
        }
    }

    fun toggle(): Boolean { enabled = !enabled; return enabled }

    fun release() {
        tone?.release()
        thread.quitSafely()
    }
}

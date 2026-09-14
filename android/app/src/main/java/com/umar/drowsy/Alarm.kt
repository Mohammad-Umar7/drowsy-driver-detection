package com.umar.drowsy

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * The sound design, shared with src/alarm.py and kept in step by
 * scripts/check_port_parity.py.
 */
object AlarmCfg {
    const val SAMPLE_RATE = 22050
    const val SIREN_LOW_HZ = 700.0
    const val SIREN_HIGH_HZ = 1500.0
    const val SIREN_SWEEP_MS = 450
    const val DROWSY_HI_HZ = 880.0
    const val DROWSY_LO_HZ = 660.0
    const val DROWSY_NOTE_MS = 180
    const val DISTRACT_LO_HZ = 600.0
    const val DISTRACT_HI_HZ = 800.0
    const val DISTRACT_CHIRP_MS = 90
    const val CRITICAL_REPEATS_MIN = 2
    const val CRITICAL_REPEATS_MAX = 4
    const val ESCALATION_WINDOW_SEC = 12.0
}

/**
 * Synthesises the alarm patterns as 16-bit PCM. Same maths as alarm.py:
 * a pitch sweep with harmonics for harshness and an 8 ms raised-cosine ramp
 * at each end so nothing clicks.
 */
object AlarmSynth {
    private const val SR = AlarmCfg.SAMPLE_RATE
    private val harmonics = doubleArrayOf(1.0, 0.5, 0.25)

    // The true peak of the harmonic mix over one cycle. Dividing by the sum
    // of the weights (1.75) does not give full scale - the harmonics never
    // all peak at the same instant - and the siren came out at 0.8 of what
    // the speaker can do. Loudness is the point, so normalise by the real peak.
    private val mixPeak: Double = run {
        var peak = 0.0
        for (i in 0 until 8192) {
            val p = 2.0 * PI * i / 8191
            var x = 0.0
            for (k in harmonics.indices) x += harmonics[k] * sin((k + 1) * p)
            peak = max(peak, kotlin.math.abs(x))
        }
        peak
    }

    fun tone(f0: Double, f1: Double, ms: Int, gain: Double): FloatArray {
        val n = max(1, SR * ms / 1000)
        val out = FloatArray(n)
        val duration = n.toDouble() / SR
        val ramp = max(2, (SR * 0.008).toInt())
        val hsum = mixPeak
        var phase = 0.0
        for (i in 0 until n) {
            val t = i.toDouble() / SR
            // The phase is the INTEGRAL of the frequency. f*t is wrong for a
            // sweep and ends at twice the intended pitch.
            val f = f0 + (f1 - f0) * (t / duration)
            phase += 2.0 * PI * f / SR
            var x = 0.0
            for (k in harmonics.indices) x += harmonics[k] * sin((k + 1) * phase)
            x /= hsum
            var env = 1.0
            if (n >= 2 * ramp) {
                if (i < ramp) env = 0.5 - 0.5 * cos(PI * i / (ramp - 1.0))
                else if (i >= n - ramp) env = 0.5 - 0.5 * cos(PI * (n - 1 - i) / (ramp - 1.0))
            }
            out[i] = (x * env * gain).toFloat()
        }
        return out
    }

    fun silence(ms: Int) = FloatArray(SR * ms / 1000)

    /** The whole pattern for one alarm, as PCM16. */
    fun render(kind: String, level: Int): ShortArray {
        val parts = ArrayList<FloatArray>()
        when (kind) {
            "distract" -> {
                val chirp = tone(AlarmCfg.DISTRACT_LO_HZ, AlarmCfg.DISTRACT_HI_HZ,
                    AlarmCfg.DISTRACT_CHIRP_MS, 0.35)
                parts += chirp; parts += silence(70); parts += chirp
            }
            "critical" -> {
                val repeats = min(AlarmCfg.CRITICAL_REPEATS_MAX,
                    AlarmCfg.CRITICAL_REPEATS_MIN + max(0, level))
                val up = tone(AlarmCfg.SIREN_LOW_HZ, AlarmCfg.SIREN_HIGH_HZ, AlarmCfg.SIREN_SWEEP_MS, 1.0)
                val down = tone(AlarmCfg.SIREN_HIGH_HZ, AlarmCfg.SIREN_LOW_HZ, AlarmCfg.SIREN_SWEEP_MS, 1.0)
                for (i in 0 until repeats) {
                    if (i > 0) parts += silence(40)
                    parts += up; parts += down
                }
            }
            else -> {
                val hi = tone(AlarmCfg.DROWSY_HI_HZ, AlarmCfg.DROWSY_HI_HZ, AlarmCfg.DROWSY_NOTE_MS, 0.75)
                val lo = tone(AlarmCfg.DROWSY_LO_HZ, AlarmCfg.DROWSY_LO_HZ, AlarmCfg.DROWSY_NOTE_MS, 0.75)
                parts += hi; parts += lo; parts += silence(60); parts += hi; parts += lo
            }
        }
        val total = parts.sumOf { it.size }
        val pcm = ShortArray(total)
        var i = 0
        for (p in parts) for (v in p) {
            pcm[i++] = (v.coerceIn(-1f, 1f) * 32767f).roundToInt().toShort()
        }
        return pcm
    }
}

/**
 * Escalating audible + haptic + spoken alarm.
 *
 * WHAT WAKES A PERSON UP. Not a beep - the first version played fixed-pitch
 * ToneGenerator beeps, the sound of a microwave, and a driver who is
 * genuinely falling asleep sleeps straight through those. Three things work:
 *
 *   1. A SIREN, not a tone. A pitch sweeping 700 -> 1500 Hz and back cannot
 *      be habituated to the way a steady note can, and the harmonics make it
 *      harsh rather than pure.
 *   2. ESCALATION. Every consecutive CRITICAL alarm inside a 12 s window is
 *      longer than the last: two sweeps, then three, then four.
 *   3. A VOICE. After the siren the phone SAYS "Wake up" - and once it has
 *      escalated, "Pull over safely". Speech tells a half-awake person what
 *      to do; a tone cannot.
 *
 * On top of that, for CRITICAL the alarm stream is raised to at least 80% of
 * maximum for the duration and restored afterwards, and the phone vibrates
 * hard. Everything plays on USAGE_ALARM, so the ringer switch cannot silence
 * it - a drowsiness warning that "silent mode" can mute is not a safety
 * feature.
 *
 * THE RULE: never block the camera thread. Everything here is queued onto a
 * dedicated background thread and the caller returns immediately.
 */
class Alarm(context: Context) {

    var enabled = true

    private val thread = HandlerThread("alarm").apply { start() }
    private val handler = Handler(thread.looper)
    private val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    private val audio = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
    private val attrs = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ALARM)
        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
        .build()

    // Text to speech is initialised asynchronously and may take a second, or
    // be missing on a stripped-down device. The siren never depends on it.
    @Volatile private var ttsReady = false
    private var tts: TextToSpeech? = null

    @Volatile private var playing = false
    @Volatile private var preempt = false          // cut the current pattern short
    @Volatile private var current: AudioTrack? = null
    @Volatile private var currentKind = ""
    private var gen = 0                            // which post owns `playing`
    private var criticalLevel = 0
    private var lastCriticalT = Double.NaN
    private val cache = HashMap<String, ShortArray>()

    /** What was last played: (kind, escalation level). */
    @Volatile var last: Pair<String, Int>? = null
        private set

    init {
        try {
            tts = TextToSpeech(context.applicationContext) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts?.setAudioAttributes(attrs)
                    tts?.language = Locale.US
                    ttsReady = true
                } else {
                    Log.w(TAG, "text to speech unavailable ($status); siren only")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "text to speech init failed; siren only", e)
        }
    }

    private fun escalate(now: Double): Int {
        val recent = !lastCriticalT.isNaN() && now - lastCriticalT <= AlarmCfg.ESCALATION_WINDOW_SEC
        criticalLevel = if (recent) min(AlarmCfg.CRITICAL_REPEATS_MAX - AlarmCfg.CRITICAL_REPEATS_MIN,
            criticalLevel + 1) else 0
        lastCriticalT = now
        return criticalLevel
    }

    private fun pcm(kind: String, level: Int): ShortArray =
        cache.getOrPut("$kind/$level") { AlarmSynth.render(kind, level) }

    /**
     * Start the pattern for `kind` ('distract' | 'drowsy' | 'critical').
     *
     * A fire() that lands while a pattern is playing is dropped - the driver
     * is already hearing it - with ONE exception: a CRITICAL siren cuts a
     * lesser pattern short and starts at once. The drowsy nudge and the
     * microsleep it warns about are often a second apart, and the siren
     * must not queue behind two polite beeps. Critical escalation is counted
     * either way, so the NEXT burst is the bigger one.
     */
    fun fire(kind: String, now: Double = System.nanoTime() / 1e9): Boolean {
        if (!enabled) return false
        val k = if (kind == "distract" || kind == "critical") kind else "drowsy"
        val level = if (k == "critical") escalate(now) else 0
        val myGen: Int
        synchronized(this) {
            if (playing) {
                if (!(k == "critical" && currentKind != "critical")) return false
                preempt = true
                try { current?.stop() } catch (e: Exception) { }
            }
            gen += 1; myGen = gen
            playing = true
            currentKind = k
        }
        last = k to level
        return post(myGen) {
            val restore = if (k == "critical") boostVolume() else null
            try {
                play(pcm(k, level))
            } finally {
                restore?.let { restoreVolume(it) }
            }
            if (preempt) return@post
            when (k) {
                "critical" -> {
                    speak(if (level > 0) "Wake up! Pull over safely." else "Wake up!")
                    // Road noise can bury a sound; a phone shaking in its
                    // mount is still felt and seen.
                    vibrate(longArrayOf(0, 500, 150, 500, 150, 700))
                }
                "drowsy" -> vibrate(longArrayOf(0, 250))
            }
        }
    }

    /**
     * Play the critical pattern once, without touching the escalation state,
     * so the driver can hear what it sounds like and check the volume before
     * relying on it. Long-press Mute in the app; works while muted, too.
     */
    fun test(): Boolean {
        val myGen: Int
        synchronized(this) {
            if (playing) return false
            gen += 1; myGen = gen
            playing = true
            currentKind = "critical"
        }
        return post(myGen) {
            play(pcm("critical", 0))
            speak("This is the alarm.")
        }
    }

    /** Queue `body` on the alarm thread; the flag is released by whoever owns it. */
    private fun post(myGen: Int, body: () -> Unit): Boolean {
        val posted = handler.post {
            preempt = false
            try {
                body()
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            } catch (e: Exception) {
                // Audio must never take the detector down with it.
            } finally {
                // Only the newest post owns the flag: a pre-empted pattern
                // finishing late must not clear it under the siren.
                synchronized(this) { if (gen == myGen) playing = false }
            }
        }
        // post() returns false once the looper has quit (teardown). Without
        // this the flag would stay set and the alarm would be dead for good.
        if (!posted) synchronized(this) { if (gen == myGen) playing = false }
        return posted
    }

    private fun play(pcm: ShortArray) {
        val track = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(AlarmCfg.SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(pcm.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        try {
            current = track
            track.write(pcm, 0, pcm.size)
            track.play()
            // Sleep in slices so a pre-empting siren can cut this short.
            val total = pcm.size * 1000L / AlarmCfg.SAMPLE_RATE + 60
            var slept = 0L
            while (slept < total && !preempt) {
                Thread.sleep(50)
                slept += 50
            }
            try { track.stop() } catch (e: IllegalStateException) { }
        } finally {
            current = null
            track.release()
        }
    }

    /** Raise the alarm stream to at least 80% for a critical burst. Returns the level to restore. */
    private fun boostVolume(): Int? {
        val am = audio ?: return null
        return try {
            val maxV = am.getStreamMaxVolume(AudioManager.STREAM_ALARM)
            val cur = am.getStreamVolume(AudioManager.STREAM_ALARM)
            val want = (maxV * 0.8).roundToInt()
            if (cur >= want) null else { am.setStreamVolume(AudioManager.STREAM_ALARM, want, 0); cur }
        } catch (e: SecurityException) {
            null            // Do Not Disturb can forbid volume changes
        }
    }

    private fun restoreVolume(level: Int) {
        try { audio?.setStreamVolume(AudioManager.STREAM_ALARM, level, 0) } catch (e: Exception) { }
    }

    private fun speak(text: String) {
        val t = tts ?: return
        if (!ttsReady) return
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "drowsy-alarm")
        // speak() is asynchronous; hold the alarm thread roughly as long as
        // the phrase so `playing` stays true until it has been said.
        Thread.sleep(300L + 90L * text.length)
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
        enabled = false
        thread.quitSafely()
        try { tts?.stop(); tts?.shutdown() } catch (e: Exception) { }
    }

    companion object {
        private const val TAG = "Drowsy"
    }
}

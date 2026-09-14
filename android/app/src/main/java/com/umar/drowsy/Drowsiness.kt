package com.umar.drowsy

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Kotlin port of src/config.py (DrowsyCfg) and src/drowsiness.py.
 *
 * Every threshold here is the value the desktop version validated, kept in one
 * object so the two implementations can be diffed at a glance.
 */
object Cfg {
    // eye closure
    const val EAR_THRESH = 0.21
    const val EAR_CALIB_RATIO = 0.75
    const val CNN_CLOSED_THRESH = 0.342      // chosen by evaluate.py, not 0.5
    const val FUSION_CNN_WEIGHT = 0.65

    // eye visibility
    const val EYE_VIS_MIN_RATIO = 0.62
    const val EYE_MIN_WIDTH_PX = 15.0

    // blink / microsleep
    const val MICROSLEEP_SEC = 2.0
    const val BLINK_MAX_SEC = 0.45
    // Closures between a blink and a microsleep: blink duration climbing is
    // one of the earliest fatigue signs, and was previously dropped.
    const val LONG_BLINK_RATE_WARN = 3

    // PERCLOS
    const val PERCLOS_WINDOW_SEC = 30.0
    const val PERCLOS_WARN = 0.15
    const val PERCLOS_CRITICAL = 0.30
    const val PERCLOS_MIN_OBS_SEC = 12.0

    // yawning
    const val MAR_THRESH = 0.60
    const val YAWN_MIN_SEC = 1.5
    const val YAWN_WINDOW_SEC = 60.0
    const val YAWN_RATE_WARN = 3

    // head pose
    const val PITCH_NOD_DEG = -18.0
    const val NOD_MIN_SEC = 2.5
    const val YAW_DISTRACT_DEG = 35.0
    const val DISTRACT_MIN_SEC = 2.5

    // alarm
    const val ALARM_COOLDOWN_SEC = 4.0
    const val DISTRACT_ALARM_COOLDOWN_SEC = 9.0
    const val FACE_LOST_GRACE_SEC = 2.0
    // Hold DROWSY/CRITICAL and keep alarming for this long after the face
    // vanishes. A drowsy driver slumping out of frame must not silence it.
    const val FACE_LOST_HOLD_SEC = 15.0
    // An otherwise-awake driver out of view this long gets a gentle nudge.
    const val FACE_LOST_NUDGE_SEC = 8.0
}

enum class Level(val rank: Int, val text: String) {
    NO_FACE(0, "NO FACE"),
    AWAKE(1, "AWAKE"),
    DISTRACTED(2, "LOOK AT THE ROAD"),
    DROWSY(3, "DROWSY"),
    CRITICAL(4, "WAKE UP!");
}

data class DrowsyState(
    val level: Level = Level.AWAKE,
    val closed: Boolean = false,
    val closedScore: Double = 0.0,
    val perclos: Double = 0.0,
    val closureSec: Double = 0.0,
    val microsleep: Boolean = false,
    val blinks: Int = 0,
    val longBlinks: Int = 0,
    val longBlinkRate: Double = 0.0,
    val yawns: Int = 0,
    val yawning: Boolean = false,
    val nodding: Boolean = false,
    val distracted: Boolean = false,
    val eyesReliable: Boolean = true,
    val reasons: List<String> = listOf("normal"),
    val shouldAlarm: Boolean = false,
    val alarmKind: String = ""
)

/**
 * The temporal brain. Identical logic to src/drowsiness.py.
 *
 *     A closed eye is NOT drowsiness. A closed eye held two seconds IS.
 *
 * Everything is measured in SECONDS, never frames. A phone throttling under
 * heat drops from 30 fps to 12, and any threshold expressed in frames would
 * silently change meaning; timestamps make the behaviour identical at any
 * frame rate.
 */
class DrowsinessMonitor(
    var earThresh: Double = Cfg.EAR_THRESH,
    private val cnnThresh: Double = Cfg.CNN_CLOSED_THRESH,
    private val cnnWeight: Double = Cfg.FUSION_CNN_WEIGHT
) {
    private class Sample(val t: Double, val closed: Boolean, val dt: Double)

    private val window = ArrayDeque<Sample>()
    private val yawnTimes = ArrayDeque<Double>()
    private val blinkTimes = ArrayDeque<Double>()
    private val longBlinkTimes = ArrayDeque<Double>()

    private var lastT: Double? = null
    private var closedSince: Double? = null
    private var yawnSince: Double? = null
    private var nodSince: Double? = null
    private var distractSince: Double? = null
    private var lastFaceT: Double? = null
    private var lostSince: Double? = null
    private var levelAtLoss: Level = Level.AWAKE
    private var lastAlarmT = 0.0
    private var lastDistractAlarmT = 0.0

    var blinks = 0; private set
    var longBlinks = 0; private set
    var yawns = 0; private set
    var state = DrowsyState(); private set

    fun reset() {
        window.clear(); yawnTimes.clear(); blinkTimes.clear(); longBlinkTimes.clear()
        lastT = null; closedSince = null; yawnSince = null
        nodSince = null; distractSince = null; lastFaceT = null
        lostSince = null; levelAtLoss = Level.AWAKE
        lastAlarmT = 0.0; lastDistractAlarmT = 0.0
        blinks = 0; longBlinks = 0; yawns = 0
        state = DrowsyState()
    }

    /**
     * Combine the CNN probability with geometric EAR.
     *
     * They fail in different situations -- the CNN knows nothing about THIS
     * person's eye shape, EAR breaks with glasses and off-angle faces -- so
     * mixing them is more robust than either alone. EAR is first mapped onto a
     * 0..1 "closedness" that reads 0.5 exactly at the threshold, so both live
     * on the same scale before being blended.
     */
    private fun fuse(closedProb: Double?, ear: Double): Double {
        val span = max(earThresh * 0.6, 1e-6)
        val earScore = min(1.0, max(0.0, 0.5 + 0.5 * ((earThresh - ear) / span)))
        if (closedProb == null) return earScore

        val cnnScore = when {
            cnnThresh <= 0.0 || cnnThresh >= 1.0 -> closedProb
            closedProb <= cnnThresh -> 0.5 * closedProb / cnnThresh
            else -> 0.5 + 0.5 * (closedProb - cnnThresh) / (1.0 - cnnThresh)
        }
        return cnnWeight * cnnScore + (1 - cnnWeight) * earScore
    }

    private fun prune(now: Double) {
        val cut = now - Cfg.PERCLOS_WINDOW_SEC
        while (window.isNotEmpty() && window.first().t < cut) window.removeFirst()
        val cutY = now - Cfg.YAWN_WINDOW_SEC
        while (yawnTimes.isNotEmpty() && yawnTimes.first() < cutY) yawnTimes.removeFirst()
        while (blinkTimes.isNotEmpty() && blinkTimes.first() < cutY) blinkTimes.removeFirst()
        while (longBlinkTimes.isNotEmpty() && longBlinkTimes.first() < cutY) longBlinkTimes.removeFirst()
    }

    fun update(
        faceFound: Boolean,
        closedProb: Double? = null,
        ear: Double = 0.3,
        mar: Double = 0.0,
        pitch: Double = 0.0,
        yaw: Double = 0.0,
        eyesReliable: Boolean = true,
        now: Double
    ): DrowsyState {
        val dt = lastT?.let { max(0.0, min(1.0, now - it)) } ?: 0.0
        lastT = now

        // ---- no face ----
        if (!faceFound) {
            val lf = lastFaceT
            if (lf != null && now - lf < Cfg.FACE_LOST_GRACE_SEC) {
                // A hand passing the lens should not reset everything. Clear
                // the alarm edge so a stale true cannot re-fire every frame.
                state = state.copy(reasons = listOf("face lost (grace)"),
                    shouldAlarm = false, alarmKind = "")
                return state
            }
            // Genuinely gone. Freeze the closure timer AND the mouth/head
            // timers: a yawn or nod that began before the dropout must not be
            // "completed" the instant the face returns.
            val seenBefore = lastFaceT != null
            if (lostSince == null) {
                lostSince = lastFaceT ?: now
                levelAtLoss = state.level
            }
            val gone = now - lostSince!!
            closedSince = null; yawnSince = null; nodSince = null; distractSince = null

            var level = Level.NO_FACE
            var reasons = listOf("no face detected")
            var shouldAlarm = false
            var alarmKind = ""

            if (levelAtLoss.rank >= Level.DROWSY.rank && gone <= Cfg.FACE_LOST_HOLD_SEC) {
                // THE SAFETY CASE. A drowsy driver slumping out of frame is the
                // most dangerous way to lose a face. NO_FACE ranks below AWAKE
                // and never alarms, so the old code went SILENT at exactly the
                // moment it mattered most. Hold the level and keep alarming.
                level = levelAtLoss
                reasons = listOf("FACE LOST while ${level.text} (%.0fs)".format(gone))
                if (now - lastAlarmT >= Cfg.ALARM_COOLDOWN_SEC) {
                    shouldAlarm = true
                    alarmKind = if (level == Level.CRITICAL) "critical" else "drowsy"
                    lastAlarmT = now
                }
            } else if (seenBefore && gone >= Cfg.FACE_LOST_NUDGE_SEC) {
                // Not drowsy, but out of view for a while: a gentle nudge.
                // Only once a driver has actually been seen - the app has
                // just been opened and the phone is still going into its
                // mount, and beeping at an empty seat every 9 s is exactly
                // the nagging that gets the whole thing switched off.
                level = Level.DISTRACTED
                reasons = listOf("driver not visible (%.0fs)".format(gone))
                if (now - lastDistractAlarmT >= Cfg.DISTRACT_ALARM_COOLDOWN_SEC) {
                    shouldAlarm = true; alarmKind = "distract"; lastDistractAlarmT = now
                }
            }

            state = DrowsyState(level = level, blinks = blinks, longBlinks = longBlinks,
                yawns = yawns, reasons = reasons, shouldAlarm = shouldAlarm,
                alarmKind = alarmKind, eyesReliable = false)
            return state
        }

        lastFaceT = now
        lostSince = null
        prune(now)

        // ---- 1. closed right now? ----
        // With the head turned far enough that no eye is properly visible the
        // incoming numbers are nonsense (a 4 px eye returned EAR 1.05 in real
        // testing). Abstaining beats guessing: PERCLOS stops accumulating and
        // the closure timer resets, so a head turn can never look like sleep.
        val score: Double
        val closed: Boolean
        if (!eyesReliable) {
            closedSince = null
            score = 0.0; closed = false
        } else {
            score = fuse(closedProb, ear)
            closed = score > 0.5
            window.addLast(Sample(now, closed, dt))
        }

        // ---- 2. PERCLOS, weighted by real elapsed time ----
        var tot = 0.0; var cls = 0.0
        for (s in window) { tot += s.dt; if (s.closed) cls += s.dt }
        val perclos = if (tot > 0.5) cls / tot else 0.0
        val perclosReady = tot >= Cfg.PERCLOS_MIN_OBS_SEC

        // ---- 3. blink vs microsleep ----
        var microsleep = false
        if (closed) {
            if (closedSince == null) closedSince = now
            if (now - closedSince!! >= Cfg.MICROSLEEP_SEC) microsleep = true
        } else {
            closedSince?.let {
                val len = now - it
                if (len <= Cfg.BLINK_MAX_SEC) { blinks++; blinkTimes.addLast(now) }
                // Too long for a blink, too short for a microsleep: previously
                // this fell through both branches and was forgotten.
                else if (len < Cfg.MICROSLEEP_SEC) { longBlinks++; longBlinkTimes.addLast(now) }
            }
            closedSince = null
        }
        val closureSec = closedSince?.let { now - it } ?: 0.0

        // ---- 4. yawning: must be BIG and SUSTAINED, which excludes talking ----
        var yawning = false
        if (mar >= Cfg.MAR_THRESH) {
            if (yawnSince == null) yawnSince = now
            else if (now - yawnSince!! >= Cfg.YAWN_MIN_SEC) yawning = true
        } else {
            yawnSince?.let {
                if (now - it >= Cfg.YAWN_MIN_SEC) { yawns++; yawnTimes.addLast(now) }
            }
            yawnSince = null
        }

        // ---- 5. head pose ----
        var nodding = false
        if (pitch <= Cfg.PITCH_NOD_DEG) {
            if (nodSince == null) nodSince = now
            else if (now - nodSince!! >= Cfg.NOD_MIN_SEC) nodding = true
        } else nodSince = null

        var distracted = false
        if (abs(yaw) >= Cfg.YAW_DISTRACT_DEG) {
            if (distractSince == null) distractSince = now
            else if (now - distractSince!! >= Cfg.DISTRACT_MIN_SEC) distracted = true
        } else distractSince = null

        // ---- 6. decide ----
        val reasons = mutableListOf<String>()
        var level = Level.AWAKE
        fun raise(l: Level) { if (l.rank > level.rank) level = l }

        if (!eyesReliable) reasons.add("eyes not visible (head turned)")

        if (microsleep) { level = Level.CRITICAL; reasons.add("MICROSLEEP %.1fs".format(closureSec)) }

        // Longer than a blink, so a blink during recovery cannot re-trigger.
        val closedNow = closed && closureSec >= Cfg.BLINK_MAX_SEC

        if (!perclosReady) {
            reasons.add("PERCLOS warming up (%.0f/%.0fs)".format(tot, Cfg.PERCLOS_MIN_OBS_SEC))
        } else if (perclos >= Cfg.PERCLOS_CRITICAL) {
            // PERCLOS is a 30-second AVERAGE, so it stays high for up to 30 s
            // after the driver is wide awake again. Shouting WAKE UP at someone
            // whose eyes are visibly open is how a safety device gets switched
            // off, so CRITICAL also requires the eyes to be shut right now.
            if (closedNow) { raise(Level.CRITICAL); reasons.add("PERCLOS %.0f%% (critical)".format(perclos * 100)) }
            else { raise(Level.DROWSY); reasons.add("PERCLOS %.0f%% (recovering)".format(perclos * 100)) }
        } else if (perclos >= Cfg.PERCLOS_WARN) {
            raise(Level.DROWSY); reasons.add("PERCLOS %.0f%%".format(perclos * 100))
        }

        if (yawnTimes.size >= Cfg.YAWN_RATE_WARN) {
            raise(Level.DROWSY); reasons.add("${yawnTimes.size} yawns/min")
        }
        val nLong = longBlinkTimes.size
        if (nLong >= Cfg.LONG_BLINK_RATE_WARN) {
            raise(Level.DROWSY); reasons.add("$nLong long blinks/min")
        }
        if (nodding) { raise(Level.DROWSY); reasons.add("head nodding") }
        if (distracted && level.rank < Level.DROWSY.rank) {
            raise(Level.DISTRACTED); reasons.add("looking away (%+.0f deg)".format(yaw))
        }

        // ---- 7. alarm, two channels with different cadence ----
        var shouldAlarm = false
        var alarmKind = ""
        if (level.rank >= Level.DROWSY.rank) {
            if (now - lastAlarmT >= Cfg.ALARM_COOLDOWN_SEC) {
                shouldAlarm = true
                alarmKind = if (level == Level.CRITICAL) "critical" else "drowsy"
                lastAlarmT = now
            }
        } else if (level == Level.DISTRACTED) {
            if (now - lastDistractAlarmT >= Cfg.DISTRACT_ALARM_COOLDOWN_SEC) {
                shouldAlarm = true; alarmKind = "distract"; lastDistractAlarmT = now
            }
        }

        state = DrowsyState(
            level = level, closed = closed, closedScore = score, perclos = perclos,
            closureSec = closureSec, microsleep = microsleep,
            blinks = blinks, yawns = yawns, yawning = yawning,
            longBlinks = longBlinks,
            longBlinkRate = nLong * 60.0 / Cfg.YAWN_WINDOW_SEC,
            nodding = nodding, distracted = distracted, eyesReliable = eyesReliable,
            reasons = if (reasons.isEmpty()) listOf("normal") else reasons,
            shouldAlarm = shouldAlarm, alarmKind = alarmKind
        )
        return state
    }
}

/**
 * Measures THIS person's open-eye EAR and sets their threshold from it.
 *
 * Necessary because EAR is a ratio of eye height to width and that ratio
 * genuinely differs between people: a measured baseline of 0.181 sits BELOW
 * the 0.210 default, so a fixed threshold would report that person as asleep
 * permanently.
 *
 * The clock starts on the first SAMPLE, not on start(), so an empty frame just
 * means calibration has not begun rather than that it failed. Completion needs
 * both a minimum duration and a minimum count, so stray frames cannot end it.
 */
class EarCalibrator(
    val seconds: Double = 3.0,
    private val minSamples: Int = 30,
    private val ratio: Double = Cfg.EAR_CALIB_RATIO
) {
    private val samples = mutableListOf<Double>()
    private var t0: Double? = null
    var armed = false; private set
    var failedReason = ""; private set

    fun start() { samples.clear(); t0 = null; armed = true; failedReason = "" }

    fun remaining(now: Double): Double =
        if (!armed) 0.0 else t0?.let { max(0.0, seconds - (now - it)) } ?: seconds

    /** Returns the new threshold once enough data exists, else null. */
    fun feed(ear: Double, now: Double): Double? {
        if (!armed) return null
        if (t0 == null) t0 = now
        samples.add(ear)
        if (now - t0!! < seconds) return null
        if (samples.size < minSamples) return null

        armed = false
        val med = samples.sorted()[samples.size / 2]
        // A median that looks like a shut eye means they blinked or looked
        // away. Calibrating to that sets a threshold so low the detector could
        // never fire again -- far worse than keeping the default.
        if (med < 0.10) {
            failedReason = "measured EAR %.3f looks like a CLOSED eye - retry with eyes open".format(med)
            return null
        }
        return med * ratio
    }
}

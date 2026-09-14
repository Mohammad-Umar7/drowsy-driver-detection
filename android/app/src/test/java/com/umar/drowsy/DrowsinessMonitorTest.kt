package com.umar.drowsy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * The Python state machine is covered by tests/test_drowsiness.py. This is
 * the same set of scenarios run against the Kotlin port, on the plain JVM -
 * Drowsiness.kt has no Android dependency, so no device or emulator is
 * needed:
 *
 *     ./gradlew testDebugUnitTest
 *
 * check_port_parity.py catches a constant that drifts. It cannot catch a
 * branch that drifts, and until this file existed nothing did: the phone's
 * temporal logic had never been executed by a test at all. Every scenario
 * here feeds simulated timestamps, so a 60-second drive takes milliseconds.
 */
class DrowsinessMonitorTest {

    private val fps = 30.0
    private val dt = 1.0 / fps
    private val open = 0.30
    private val shut = 0.08

    /** Feed `seconds` of identical frames at 30 fps. Returns (state, end time). */
    private fun run(
        m: DrowsinessMonitor, seconds: Double, t0: Double, ear: Double,
        mar: Double = 0.1, pitch: Double = 0.0, yaw: Double = 0.0,
        closedProb: Double? = null, reliable: Boolean = true
    ): Pair<DrowsyState, Double> {
        var st = m.state
        val n = (seconds * fps).toInt()
        for (i in 0 until n) {
            st = m.update(faceFound = true, closedProb = closedProb, ear = ear, mar = mar,
                pitch = pitch, yaw = yaw, eyesReliable = reliable, now = t0 + i * dt)
        }
        return st to (t0 + n * dt)
    }

    @Test fun eyesOpenStaysAwake() {
        val m = DrowsinessMonitor()
        val (st, _) = run(m, 20.0, 1000.0, ear = open)
        assertEquals(Level.AWAKE, st.level)
        assertTrue("perclos ${st.perclos}", st.perclos < 0.01)
        assertFalse(st.closed)
    }

    @Test fun normalBlinkingDoesNotAlarm() {
        val m = DrowsinessMonitor()
        var t = 1000.0
        repeat(8) {
            t = run(m, 3.8, t, ear = open).second
            t = run(m, 0.2, t, ear = shut).second
        }
        assertEquals(Level.AWAKE, m.state.level)
        assertTrue("perclos ${m.state.perclos}", m.state.perclos < Cfg.PERCLOS_WARN)
        assertTrue("blinks ${m.state.blinks}", m.state.blinks >= 6)
    }

    @Test fun microsleepIsCritical() {
        val m = DrowsinessMonitor()
        val (_, t) = run(m, 5.0, 1000.0, ear = open)
        val (st, _) = run(m, Cfg.MICROSLEEP_SEC + 0.4, t, ear = shut)
        assertTrue(st.microsleep)
        assertEquals(Level.CRITICAL, st.level)
        assertTrue(st.reasons.any { it.contains("MICROSLEEP") })
        assertEquals("a microsleep is not a blink", 0, st.blinks)
    }

    @Test fun slowSlideRaisesPerclos() {
        val m = DrowsinessMonitor()
        var t = 1000.0
        repeat(20) {
            t = run(m, 1.0, t, ear = open).second
            t = run(m, 0.5, t, ear = shut).second
        }
        assertTrue("perclos ${m.state.perclos}", m.state.perclos >= Cfg.PERCLOS_WARN)
        assertTrue(m.state.level.rank >= Level.DROWSY.rank)
    }

    @Test fun threeYawnsAreDrowsy() {
        val m = DrowsinessMonitor()
        var t = 1000.0
        repeat(3) {
            t = run(m, Cfg.YAWN_MIN_SEC + 0.6, t, ear = open, mar = 0.75).second
            t = run(m, 4.0, t, ear = open, mar = 0.10).second
        }
        assertEquals(3, m.state.yawns)
        assertTrue(m.state.level.rank >= Level.DROWSY.rank)
    }

    @Test fun talkingIsNotYawning() {
        val m = DrowsinessMonitor()
        var t = 1000.0
        repeat(15) {
            t = run(m, 0.3, t, ear = open, mar = 0.70).second
            t = run(m, 0.4, t, ear = open, mar = 0.15).second
        }
        assertEquals(0, m.state.yawns)
    }

    @Test fun headNodIsDrowsy() {
        val m = DrowsinessMonitor()
        val (_, t) = run(m, 3.0, 1000.0, ear = open)
        val (st, _) = run(m, Cfg.NOD_MIN_SEC + 0.5, t, ear = open, pitch = -25.0)
        assertTrue(st.nodding)
        assertTrue(st.level.rank >= Level.DROWSY.rank)
    }

    @Test fun lookingAwayIsDistractedNotDrowsy() {
        val m = DrowsinessMonitor()
        val (_, t) = run(m, 3.0, 1000.0, ear = open)
        val (st, _) = run(m, Cfg.DISTRACT_MIN_SEC + 0.5, t, ear = open, yaw = 45.0)
        assertTrue(st.distracted)
        assertEquals(Level.DISTRACTED, st.level)
    }

    @Test fun verdictIsIndependentOfFrameRate() {
        val secs = Cfg.MICROSLEEP_SEC + 0.3
        val results = listOf(10, 30, 60).map { rate ->
            val m = DrowsinessMonitor()
            val step = 1.0 / rate
            var t = 1000.0
            for (i in 0 until 5 * rate) m.update(faceFound = true, ear = open, now = t + i * step)
            t += 5.0
            var st = m.state
            for (i in 0 until (secs * rate).toInt()) st = m.update(faceFound = true, ear = shut, now = t + i * step)
            st.level
        }
        assertEquals(1, results.toSet().size)
        assertTrue(results.all { it == Level.CRITICAL })
    }

    @Test fun cnnOutvotesEar() {
        val m = DrowsinessMonitor()
        assertTrue(m.update(faceFound = true, closedProb = 0.99, ear = shut, now = 1000.0).closed)
        m.reset()
        assertFalse(m.update(faceFound = true, closedProb = 0.01, ear = open, now = 1000.0).closed)
        m.reset()
        // The CNN carries 65% of the weight, so it wins a disagreement.
        assertTrue(m.update(faceFound = true, closedProb = 0.97, ear = open, now = 1000.0).closed)
    }

    @Test fun perclosNeedsEnoughDataFirst() {
        val m = DrowsinessMonitor()
        val (_, t) = run(m, 3.0, 1000.0, ear = open)
        val (st, _) = run(m, 1.5, t, ear = shut)
        assertTrue("perclos ${st.perclos}", st.perclos > 0.25)
        assertTrue(st.reasons.none { it.contains("PERCLOS") && !it.contains("warming") })
    }

    @Test fun criticalReleasesWhenEyesReopen() {
        val m = DrowsinessMonitor()
        var (st, t) = run(m, 15.0, 1000.0, ear = shut)
        assertEquals(Level.CRITICAL, st.level)
        run(m, 5.0, t, ear = open).let { st = it.first; t = it.second }
        assertTrue("perclos ${st.perclos}", st.perclos > 0.5)
        assertEquals("still high PERCLOS, eyes open", Level.DROWSY, st.level)
        assertTrue(st.reasons.any { it.contains("recovering") })
        // A blink during recovery must not re-trigger.
        run(m, 0.2, t, ear = shut).let { st = it.first; t = it.second }
        assertTrue(st.level != Level.CRITICAL)
        // Closing them again must scream again.
        st = run(m, Cfg.MICROSLEEP_SEC + 0.5, t, ear = shut).first
        assertEquals(Level.CRITICAL, st.level)
    }

    @Test fun headTurnIsNeverAClosure() {
        val m = DrowsinessMonitor()
        var (st, t) = run(m, 6.0, 1000.0, ear = open)
        // The garbage a foreshortened eye produces, measured on a real turn.
        for (i in 0 until (5 * fps).toInt()) {
            st = m.update(faceFound = true, closedProb = 0.68, ear = 1.05, yaw = 70.0,
                eyesReliable = false, now = t + i * dt)
        }
        t += 5.0
        assertFalse(st.closed)
        assertFalse(st.microsleep)
        assertEquals(Level.DISTRACTED, st.level)
        assertTrue("perclos ${st.perclos}", st.perclos < 0.05)
        // Facing forward again with shut eyes must still fire.
        st = run(m, Cfg.MICROSLEEP_SEC + 0.4, t, ear = shut).first
        assertEquals(Level.CRITICAL, st.level)
    }

    @Test fun distractionAlarmIsGentleAndRare() {
        val m = DrowsinessMonitor()
        val (_, t) = run(m, 5.0, 1000.0, ear = open)
        val fired = mutableListOf<String>()
        for (i in 0 until (30 * fps).toInt()) {
            val st = m.update(faceFound = true, ear = open, yaw = 50.0, now = t + i * dt)
            if (st.shouldAlarm) fired += st.alarmKind
        }
        assertTrue(fired.isNotEmpty())
        assertTrue(fired.all { it == "distract" })
        assertTrue("fired ${fired.size}", fired.size <= 4)
    }

    @Test fun drowsyDriverLeavingFrameKeepsAlarming() {
        val m = DrowsinessMonitor()
        var (st, t) = run(m, 5.0, 1000.0, ear = open)
        run(m, Cfg.MICROSLEEP_SEC + 0.5, t, ear = shut).let { st = it.first; t = it.second }
        assertEquals(Level.CRITICAL, st.level)
        var fired = 0
        for (i in 0 until (10 * fps).toInt()) {
            st = m.update(faceFound = false, now = t + i * dt)
            if (st.shouldAlarm) fired++
        }
        assertEquals("held, not dropped to NO_FACE", Level.CRITICAL, st.level)
        assertTrue("fired $fired", fired >= 2)
        assertTrue(st.reasons.any { it.contains("FACE LOST") })
        for (i in (10 * fps).toInt() until ((Cfg.FACE_LOST_HOLD_SEC + 4.0) * fps).toInt()) {
            st = m.update(faceFound = false, now = t + i * dt)
        }
        assertTrue("releases after the hold", st.level != Level.CRITICAL)
    }

    @Test fun awakeDriverOutOfViewGetsANudge() {
        val m = DrowsinessMonitor()
        val (_, t) = run(m, 5.0, 1000.0, ear = open)
        val kinds = mutableSetOf<String>()
        var st = m.state
        for (i in 0 until ((Cfg.FACE_LOST_NUDGE_SEC + 3.0) * fps).toInt()) {
            st = m.update(faceFound = false, now = t + i * dt)
            if (st.shouldAlarm) kinds += st.alarmKind
        }
        assertEquals(Level.DISTRACTED, st.level)
        assertEquals(setOf("distract"), kinds)
    }

    @Test fun emptySeatAtStartupIsNotNagged() {
        val m = DrowsinessMonitor()
        var fired = 0
        var st = m.state
        for (i in 0 until (Cfg.FACE_LOST_NUDGE_SEC * 3 * fps).toInt()) {
            st = m.update(faceFound = false, now = 1000.0 + i * dt)
            if (st.shouldAlarm) fired++
        }
        assertEquals(Level.NO_FACE, st.level)
        assertEquals(0, fired)
        // Once seen and gone, the nudge is earned.
        var t = 1000.0 + Cfg.FACE_LOST_NUDGE_SEC * 3
        t = run(m, 2.0, t, ear = open).second
        for (i in 0 until ((Cfg.FACE_LOST_NUDGE_SEC + 2.0) * fps).toInt()) {
            st = m.update(faceFound = false, now = t + i * dt)
            if (st.shouldAlarm) fired++
        }
        assertTrue(fired >= 1)
        assertEquals(Level.DISTRACTED, st.level)
    }

    @Test fun yawnTimerDoesNotSurviveFaceLoss() {
        val m = DrowsinessMonitor()
        var t = run(m, 3.0, 1000.0, ear = open).second
        t = run(m, 0.3, t, ear = open, mar = 0.75).second
        for (i in 0 until ((Cfg.FACE_LOST_GRACE_SEC + 1.0) * fps).toInt()) m.update(faceFound = false, now = t + i * dt)
        t += Cfg.FACE_LOST_GRACE_SEC + 1.0
        val (st, _) = run(m, 0.5, t, ear = open, mar = 0.10)
        assertEquals(0, st.yawns)
    }

    @Test fun longBlinksAreAnEarlyWarning() {
        val m = DrowsinessMonitor()
        var t = 1000.0
        repeat(3) {
            t = run(m, 5.2, t, ear = open).second
            t = run(m, 0.8, t, ear = shut).second      // > blink, < microsleep
        }
        val (st, _) = run(m, 1.0, t, ear = open)
        assertEquals(3, st.longBlinks)
        assertEquals(0, st.blinks)
        assertFalse(st.microsleep)
        assertTrue("perclos ${st.perclos}", st.perclos < Cfg.PERCLOS_WARN)
        assertEquals(Level.DROWSY, st.level)
        assertTrue(st.reasons.any { it.contains("long blinks") })
    }

    @Test fun faceLossHasAGracePeriod() {
        val m = DrowsinessMonitor()
        val (_, t) = run(m, 3.0, 1000.0, ear = open)
        assertTrue(m.update(faceFound = false, now = t + 0.5).level != Level.NO_FACE)
        assertEquals(Level.NO_FACE, m.update(faceFound = false, now = t + 5.0).level)
    }

    @Test fun calibratorWaitsForAFaceAndNeedsSamples() {
        val c = EarCalibrator(seconds = 3.0, minSamples = 30)
        c.start()
        assertTrue("armed while nobody is there", c.armed)
        assertEquals(3.0, c.remaining(500.0), 1e-9)   // clock has not started
        // Duration satisfied but only a few samples: keep waiting.
        assertNull(c.feed(0.30, 1000.0))
        assertNull(c.feed(0.30, 1004.0))
        assertTrue(c.armed)
        var out: Double? = null
        var t = 1004.0
        while (out == null && t < 1010.0) { t += 0.05; out = c.feed(0.30, t) }
        assertNotNull(out)
        assertTrue(abs(out!! - 0.30 * Cfg.EAR_CALIB_RATIO) < 1e-9)
        assertFalse(c.armed)
    }

    @Test fun calibratorRefusesShutEyes() {
        val c = EarCalibrator(seconds = 0.5, minSamples = 5)
        c.start()
        var out: Double? = null
        for (i in 0 until 40) out = c.feed(0.05, 1000.0 + i * 0.05)
        assertNull(out)
        assertFalse(c.armed)
        assertTrue(c.failedReason.contains("CLOSED"))
    }
}

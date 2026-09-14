"""
Non-blocking, escalating audible alarm.

=============================================================================
WHAT WAKES A PERSON UP
=============================================================================

A beep does not. The first version played fixed-pitch beeps through
winsound.Beep - the sound of a microwave - and a driver who is genuinely
falling asleep will sleep straight through it. Three things are known to
work, and this file does all three:

  1. A SIREN, not a tone. A pitch that sweeps 700 -> 1500 Hz and back is far
     harder to ignore than a steady note, because hearing habituates to a
     constant sound within seconds and cannot habituate to one that keeps
     changing. Harmonics are added on top of the fundamental so the sound is
     harsh rather than pure; a pure sine is the easiest sound there is to
     sleep through.

  2. ESCALATION. Every consecutive CRITICAL alarm inside a 12 s window is
     longer than the last: two sweeps, then three, then four. A driver who
     has not responded to the first burst gets a bigger one, not the same
     one again.

  3. A VOICE. After the siren, the phone or PC SAYS "Wake up" - and, once the
     alarm has already escalated, "Pull over safely". Speech carries meaning
     a tone cannot; it tells a half-awake person what to do.

Distraction stays a soft two-note chirp: it is a reminder, not an emergency,
and an aggressive tone for a mirror check would train the driver to ignore
every alert. DROWSY is a firm hi-lo double tone in between.

The waveforms are synthesised here with numpy and handed to the platform's
player as a WAV in memory, so there are no audio files to ship and no audio
library to install. The same numbers (below) drive Alarm.kt on the phone,
and scripts/check_port_parity.py keeps them in step.

THE ONE RULE: never block the video loop. Everything audible happens on a
background thread; fire() returns immediately.
"""
import io
import shutil
import subprocess
import sys
import threading
import time
import wave

import numpy as np

try:
    import winsound                    # Windows only, part of the stdlib
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

SAMPLE_RATE = 22050

# ---- the sound design, shared with Alarm.kt (checked by the parity script) --
SIREN_LOW_HZ = 700.0
SIREN_HIGH_HZ = 1500.0
SIREN_SWEEP_MS = 450            # one direction; a full cycle is up then down
DROWSY_HI_HZ = 880.0
DROWSY_LO_HZ = 660.0
DROWSY_NOTE_MS = 180
DISTRACT_LO_HZ = 600.0
DISTRACT_HI_HZ = 800.0
DISTRACT_CHIRP_MS = 90
CRITICAL_REPEATS_MIN = 2        # siren cycles on the first critical alarm
CRITICAL_REPEATS_MAX = 4        # ...and after escalating
ESCALATION_WINDOW_SEC = 12.0    # a critical alarm this soon after the last escalates


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
HARMONICS = (1.0, 0.5, 0.25)      # fundamental, 2nd, 3rd: harsh, not pure


def _mix_peak(harmonics=HARMONICS) -> float:
    """
    The true peak of the harmonic mix over one cycle.

    Dividing by the SUM of the weights (1.75) does not give full scale: the
    harmonics never all peak at the same instant, so the mix only reached
    about 0.8 and the siren was quieter than it could be. Loudness is the
    point, so normalise by the real peak instead.
    """
    p = np.linspace(0.0, 2.0 * np.pi, 8192)
    return float(np.abs(sum(a * np.sin(k * p)
                            for k, a in enumerate(harmonics, start=1))).max())


_MIX_PEAK = _mix_peak()


def _tone(f0: float, f1: float, ms: int, gain: float = 1.0,
          harmonics=HARMONICS) -> np.ndarray:
    """
    A pitch sweep from f0 to f1 over `ms` milliseconds, float32 in [-1, 1].

    The phase is the integral of the frequency, not frequency*time - the
    latter is wrong for a sweep and produces a chirp that ends at twice the
    intended pitch. An 8 ms raised-cosine ramp at each end removes the click
    that a waveform starting at full amplitude makes.
    """
    n = max(1, int(SAMPLE_RATE * ms / 1000.0))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    duration = n / SAMPLE_RATE
    freq = f0 + (f1 - f0) * (t / duration)
    phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    x = sum(a * np.sin(k * phase) for k, a in enumerate(harmonics, start=1))
    x /= _MIX_PEAK if harmonics == HARMONICS else _mix_peak(harmonics)
    ramp = max(1, int(SAMPLE_RATE * 0.008))
    env = np.ones(n)
    if n >= 2 * ramp:
        r = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp))
        env[:ramp] = r
        env[-ramp:] = r[::-1]
    return (x * env * gain).astype(np.float32)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000.0), np.float32)


def render(kind: str, level: int = 0) -> np.ndarray:
    """The whole pattern for one alarm, as float32 samples in [-1, 1]."""
    if kind == "distract":
        chirp = _tone(DISTRACT_LO_HZ, DISTRACT_HI_HZ, DISTRACT_CHIRP_MS, gain=0.35)
        parts = [chirp, _silence(70), chirp]
    elif kind == "critical":
        repeats = min(CRITICAL_REPEATS_MAX, CRITICAL_REPEATS_MIN + max(0, level))
        up = _tone(SIREN_LOW_HZ, SIREN_HIGH_HZ, SIREN_SWEEP_MS, gain=1.0)
        down = _tone(SIREN_HIGH_HZ, SIREN_LOW_HZ, SIREN_SWEEP_MS, gain=1.0)
        parts = []
        for i in range(repeats):
            if i:
                parts.append(_silence(40))
            parts += [up, down]
    else:                                   # "drowsy" and anything unknown
        hi = _tone(DROWSY_HI_HZ, DROWSY_HI_HZ, DROWSY_NOTE_MS, gain=0.75)
        lo = _tone(DROWSY_LO_HZ, DROWSY_LO_HZ, DROWSY_NOTE_MS, gain=0.75)
        parts = [hi, lo, _silence(60), hi, lo]
    return np.concatenate(parts)


def to_wav(samples: np.ndarray) -> bytes:
    """Mono 16-bit WAV bytes, ready for any player."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def duration_sec(samples: np.ndarray) -> float:
    return len(samples) / SAMPLE_RATE


# ---------------------------------------------------------------------------
# platform back ends. Each play() BLOCKS for the sound's duration, which is
# fine and even desirable: it runs on the alarm thread, never the video loop.
# ---------------------------------------------------------------------------
class _WinSound:
    """winsound fallback. PlaySound(None) from any thread stops the current sound."""

    def __call__(self, wav: bytes) -> None:
        winsound.PlaySound(wav, winsound.SND_MEMORY)

    def stop(self) -> None:
        winsound.PlaySound(None, 0)


_play_winsound = _WinSound()


class _WaveOut:
    """
    Windows playback through winmm's waveOut API, with the device held OPEN.

    winsound.PlaySound opens the output device, plays, and closes it again on
    every call. On this machine that open cost 2.5 s each time - measured: a
    quarter-second chirp took 2.8 s to "play" - so the siren started late and
    the voice after it later still. Bluetooth and power-saving audio paths
    make the open expensive; nothing about the sound does. Open the device
    once here and only ever WRITE to it, and playback starts at once.

    Pure ctypes, no dependency. Falls back to winsound if anything refuses.
    """

    WAVE_FORMAT_PCM = 1
    WAVE_MAPPER = 0xFFFFFFFF
    WHDR_DONE = 0x1

    def __init__(self, rate: int = SAMPLE_RATE):
        import ctypes
        import ctypes.wintypes as wt

        class WAVEFORMATEX(ctypes.Structure):
            _fields_ = [("wFormatTag", wt.WORD), ("nChannels", wt.WORD),
                        ("nSamplesPerSec", wt.DWORD), ("nAvgBytesPerSec", wt.DWORD),
                        ("nBlockAlign", wt.WORD), ("wBitsPerSample", wt.WORD),
                        ("cbSize", wt.WORD)]

        class WAVEHDR(ctypes.Structure):
            _fields_ = [("lpData", ctypes.c_void_p), ("dwBufferLength", wt.DWORD),
                        ("dwBytesRecorded", wt.DWORD), ("dwUser", ctypes.c_void_p),
                        ("dwFlags", wt.DWORD), ("dwLoops", wt.DWORD),
                        ("lpNext", ctypes.c_void_p), ("reserved", ctypes.c_void_p)]

        self._ctypes = ctypes
        self._WAVEHDR = WAVEHDR
        self._winmm = ctypes.windll.winmm
        self._rate = rate
        self._h = wt.HANDLE()
        fmt = WAVEFORMATEX(self.WAVE_FORMAT_PCM, 1, rate, rate * 2, 2, 16, 0)
        err = self._winmm.waveOutOpen(ctypes.byref(self._h), self.WAVE_MAPPER,
                                      ctypes.byref(fmt), 0, 0, 0)
        if err != 0:
            raise OSError(f"waveOutOpen failed with code {err}")

    def __call__(self, wav: bytes) -> None:
        with wave.open(io.BytesIO(wav), "rb") as w:
            if w.getframerate() != self._rate or w.getnchannels() != 1 \
                    or w.getsampwidth() != 2:
                _play_winsound(wav)          # not our format; the slow path
                return
            n = w.getnframes()
            pcm = w.readframes(n)
        ct = self._ctypes
        buf = ct.create_string_buffer(pcm, len(pcm))
        hdr = self._WAVEHDR()
        hdr.lpData = ct.cast(buf, ct.c_void_p)
        hdr.dwBufferLength = len(pcm)
        size = ct.sizeof(self._WAVEHDR)
        if self._winmm.waveOutPrepareHeader(self._h, ct.byref(hdr), size) != 0:
            _play_winsound(wav)
            return
        try:
            if self._winmm.waveOutWrite(self._h, ct.byref(hdr), size) != 0:
                _play_winsound(wav)
                return
            deadline = time.monotonic() + n / self._rate + 2.0
            while not (hdr.dwFlags & self.WHDR_DONE) and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            self._winmm.waveOutUnprepareHeader(self._h, ct.byref(hdr), size)

    def stop(self) -> None:
        """Abort whatever is playing; the header is marked done and __call__ returns."""
        try:
            self._winmm.waveOutReset(self._h)
        except Exception:                # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self._winmm.waveOutReset(self._h)
            self._winmm.waveOutClose(self._h)
        except Exception:                # noqa: BLE001
            pass


class _CommandPlayer:
    """afplay (macOS), paplay / aplay (Linux): they want a file, so cache one."""

    def __init__(self, argv):
        self.argv = argv
        self._files = {}
        self._proc = None

    def __call__(self, wav: bytes) -> None:
        import hashlib
        import tempfile
        key = hashlib.md5(wav).hexdigest()
        path = self._files.get(key)
        if path is None:
            f = tempfile.NamedTemporaryFile(prefix="drowsy_", suffix=".wav",
                                            delete=False)
            f.write(wav)
            f.close()
            path = self._files[key] = f.name
        self._proc = subprocess.Popen(self.argv + [path], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc = None

    def stop(self) -> None:
        p = self._proc
        if p is not None:
            try:
                p.kill()
            except Exception:            # noqa: BLE001
                pass


def _play_bell(wav: bytes) -> None:
    """Last resort: the terminal bell, held for the pattern's length."""
    print("\a", end="", flush=True)
    time.sleep(max(0.1, (len(wav) - 44) / 2 / SAMPLE_RATE))


def default_player():
    if _HAS_WINSOUND:
        try:
            return _WaveOut()
        except Exception:                # noqa: BLE001
            return _play_winsound
    for cmd in (["afplay"], ["paplay"], ["aplay", "-q"]):
        if shutil.which(cmd[0]):
            return _CommandPlayer(cmd)
    return _play_bell


class _WindowsSpeaker:
    """
    SAPI text to speech, rendered to WAV once and played like any other sound.

    Speaking through SAPI directly cost 3-5 s per phrase here: PowerShell
    start-up, loading System.Speech, then the same slow device open that
    winsound suffers from. All of that landed between the siren and the
    words, so the driver heard the alarm, then silence, then "Wake up"
    arriving late enough to be confusing.

    Rendering the phrases to WAV files takes 0.2 s, so that is done ONCE in
    the background at construction, and each phrase is then played through
    the alarm's own (already open) device. The text is always one of our own
    constants, never user input, so the quoting below is safe.
    """

    def __init__(self, player, phrases):
        self._player = player
        self._wavs = {}
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._phrases = list(phrases)
        threading.Thread(target=self._render, args=(self._phrases,),
                         daemon=True).start()

    def _render(self, phrases) -> None:
        import tempfile
        with self._lock:
            try:
                d = tempfile.mkdtemp(prefix="drowsy_voice_")
                lines = ["Add-Type -AssemblyName System.Speech",
                         "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                         "$s.Rate = 2; $s.Volume = 100",
                         "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
                         f"{SAMPLE_RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
                         "[System.Speech.AudioFormat.AudioChannel]::Mono)"]
                paths = {}
                for i, text in enumerate(phrases):
                    p = f"{d}\\v{i}.wav"
                    paths[text] = p
                    safe = text.replace("'", "")
                    lines.append(f"$s.SetOutputToWaveFile('{p}', $fmt); $s.Speak('{safe}')")
                lines.append("$s.SetOutputToNull()")
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                                "-Command", "; ".join(lines)], timeout=60,
                               creationflags=flags, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                for text, p in paths.items():
                    with open(p, "rb") as f:
                        self._wavs[text] = f.read()
            except Exception:            # noqa: BLE001
                pass                     # no voice, siren only
            finally:
                self._ready.set()

    def __call__(self, text: str) -> None:
        self._ready.wait(timeout=20.0)
        wav = self._wavs.get(text)
        if wav is None and text not in self._phrases:
            self._phrases.append(text)
            self._ready.clear()
            self._render([text])
            wav = self._wavs.get(text)
        if wav:
            self._player(wav)


def default_speaker(player=None, phrases=()):
    """
    Text to speech with whatever the OS already has; None if it has nothing.

    Windows ships SAPI, reachable from PowerShell without installing anything.
    macOS has `say`. Linux has espeak only if the user put it there.
    """
    if sys.platform == "win32":
        return _WindowsSpeaker(player or default_player(), phrases)
    if sys.platform == "darwin" and shutil.which("say"):
        return lambda text: subprocess.run(["say", "-r", "200", text], timeout=15)
    if shutil.which("espeak"):
        return lambda text: subprocess.run(["espeak", "-s", "170", text], timeout=15,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
    return None


# ---------------------------------------------------------------------------
class Alarm:
    """
    fire(kind) plays the pattern for `kind` on a background thread.

    kind is 'distract' | 'drowsy' | 'critical'. A fire() that arrives while a
    pattern is playing is dropped - the driver is already hearing it - with
    ONE exception: a CRITICAL siren cuts a lesser pattern short and starts at
    once. The drowsy nudge and the microsleep it warns about are often a
    second apart, and the siren must not queue behind two polite beeps.
    Critical escalation is counted either way, so the NEXT burst is the
    bigger one.
    """

    VOICE = {0: "Wake up!", 1: "Wake up! Pull over safely."}

    def __init__(self, enabled: bool = True, voice: bool = True,
                 player=None, speaker=None):
        self.enabled = enabled
        self.voice = voice
        self._player = player if player is not None else default_player()
        self._speak = (speaker if speaker is not None else
                       default_speaker(self._player, self.VOICE.values()))
        self._playing = False
        self._lock = threading.Lock()
        self._thread = None
        self._gen = 0                    # which play() owns the playing flag
        self._current_kind = ""
        self._critical_level = 0
        self._last_critical_t = None
        self._wav_cache = {}
        # What was last played, for tests and the session summary.
        self.last = None

    # -- escalation bookkeeping ------------------------------------------
    def _escalate(self, now: float) -> int:
        recent = (self._last_critical_t is not None and
                  now - self._last_critical_t <= ESCALATION_WINDOW_SEC)
        if recent:
            self._critical_level = min(CRITICAL_REPEATS_MAX - CRITICAL_REPEATS_MIN,
                                       self._critical_level + 1)
        else:
            self._critical_level = 0
        self._last_critical_t = now
        return self._critical_level

    def _wav(self, kind: str, level: int) -> bytes:
        key = (kind, level)
        if key not in self._wav_cache:
            self._wav_cache[key] = to_wav(render(kind, level))
        return self._wav_cache[key]

    def _play(self, kind: str, level: int, gen: int, preempt) -> None:
        try:
            if preempt is not None:
                # Cut the lesser pattern short, then wait for its thread so
                # two sounds never overlap on the device.
                stop = getattr(self._player, "stop", None)
                if stop is not None:
                    try:
                        stop()
                    except Exception:    # noqa: BLE001
                        pass
                preempt.join(timeout=2.0)
            self._player(self._wav(kind, level))
            if kind == "critical" and self.voice and self._speak is not None:
                self._speak(self.VOICE[1 if level > 0 else 0])
        except Exception:                # noqa: BLE001
            pass                         # audio must never crash the detector
        finally:
            with self._lock:
                # Only the newest play() owns the flag: a pre-empted thread
                # finishing late must not clear it under the siren.
                if self._gen == gen:
                    self._playing = False

    def fire(self, kind: str = "drowsy", critical: bool = False,
             now: float = None) -> bool:
        """Start the pattern. Returns True if a sound was started."""
        if not self.enabled:
            return False
        if critical:                     # backwards-compatible call style
            kind = "critical"
        if kind not in ("distract", "drowsy", "critical"):
            kind = "drowsy"
        level = 0
        if kind == "critical":
            level = self._escalate(time.monotonic() if now is None else now)
        with self._lock:
            preempt = None
            if self._playing:
                if not (kind == "critical" and self._current_kind != "critical"):
                    return False
                preempt = self._thread
            self._gen += 1
            gen = self._gen
            self._playing = True
            self._current_kind = kind
            t = threading.Thread(target=self._play, args=(kind, level, gen, preempt),
                                 daemon=True)
            self._thread = t
        self.last = (kind, level)
        try:
            t.start()
        except RuntimeError:
            # "can't start new thread" under resource pressure. _play never
            # runs, so its finally never clears _playing, and every later
            # fire() would return early - a silently dead alarm for the rest
            # of the session. Reset the flag so the next attempt can try again.
            with self._lock:
                if self._gen == gen:
                    self._playing = False
            return False
        return True

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def close(self) -> None:
        """Stop taking new alarms and release the audio device, if we hold one."""
        self.enabled = False
        closer = getattr(self._player, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:            # noqa: BLE001
                pass

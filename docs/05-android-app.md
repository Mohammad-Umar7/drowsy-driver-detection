# 05 — The Android app

The whole detector, running on the phone. No PC, no WiFi, no signal.

---

## Install it

Three APKs are produced. **You almost certainly want `arm64-v8a`** — every
Android phone since roughly 2017 uses that architecture.

| File | Size | For |
|---|---|---|
| `app-arm64-v8a-debug.apk` | 69 MB | virtually all modern phones |
| `app-armeabi-v7a-debug.apk` | 50 MB | older 32-bit phones |
| `app-universal-debug.apk` | 105 MB | if you are not sure — contains both |

They are built to:

```
android/app/build/outputs/apk/debug/
```

### Option A — USB cable (easiest)

1. On the phone: **Settings → About phone → tap "Build number" seven times**
   to unlock Developer options.
2. **Settings → Developer options → USB debugging → on.**
3. Plug the phone into the PC and accept the "Allow USB debugging?" prompt.
4. On the PC:

```bash
adb install -r android/app/build/outputs/apk/debug/app-arm64-v8a-debug.apk
```

### Option B — no cable

Copy the `.apk` to the phone (Google Drive, email it to yourself, USB file
transfer), open it in the phone's Files app, and allow "Install unknown apps"
when prompted.

### Rebuilding after a change

```bash
cd android
./gradlew assembleDebug
```

---

## Using it

1. Open **Drowsy Driver**. Grant camera permission.
2. It **calibrates automatically** for the first 3 seconds — look at the phone
   with your eyes open.
3. Mount the phone facing you. The front camera is used by default; **Flip**
   switches to the rear one.

| Button | Does |
|---|---|
| **Calibrate** | re-measure your open-eye EAR (do this if it misreads you) |
| **Flip** | front ↔ rear camera (stays put, and says so, if the device has no camera on the other side) |
| **Mute** | silence the alarm — a `MUTED` chip appears in the top card while it is off |
| **Reset** | clear the counters, PERCLOS window and the closure strip |

The calibrated threshold is saved, so the next launch uses it instead of
calibrating again.

The screen stays awake while the app is open, so it will not sleep mid-drive.
(`keepScreenOn` originally sat on the `<activity>` element in the manifest,
where it is not a valid attribute and was silently ignored — the phone was free
to lock in the middle of a drive. It belongs on a View, and now is.)

### The display

Top card: the state as a coloured pill that flashes on **WAKE UP!**, the reasons
behind it, and chips for frame rate and lighting condition (amber when the
light is genuinely difficult, so a bad reading can be blamed on the scene rather
than the driver). Bottom card: PERCLOS and eye-closure bars with their
threshold ticks, then the **closure strip** — the last 30 seconds of the fused
closure score, red where the eyes were shut. A blink is a narrow spike, a
microsleep is a wide red block, and the slide into fatigue is spikes getting
wider and closer together. Below that, EAR against its threshold, MAR, head
pose, which eyes are trusted and how wide they are in pixels, and the counters.

Every size is in dp/sp. The first version used pixel literals tuned on one
phone, so on a 720p screen the panels covered half the preview and on a 1440p
one the text was unreadable.

### Resolution

Frames are analysed at **1280×720**. Left to its default, CameraX's
`ImageAnalysis` delivers 640×480 — the mode the desktop version had to fight,
where an eye at driving distance is ~23 px wide and a 2 px landmark error is a
30 % error in EAR. Sensors without a 720p mode fall back to the closest one.

---

## What it does, and does not, send

The built APK requests exactly two permissions:

```
uses-permission: android.permission.CAMERA
uses-permission: android.permission.VIBRATE
```

There is **no `INTERNET` permission**. That is worth explaining, because it did
not start out that way.

The first build *did* have it. Nothing in this codebase asked for it — MediaPipe
pulls in `com.google.android.datatransport` (Google's telemetry uploader)
transitively, and Android's manifest merger silently adds a library's
permissions to your app. Inspecting the built APK is what caught it.

It is now stripped with `tools:node="remove"`. Without the INTERNET permission
the **operating system itself** refuses any socket this process opens. So
"runs entirely on device" is enforced by Android rather than asserted in a
README — which is the only version of that claim worth making.

---

## How it maps to the desktop version

Same pipeline, same thresholds, same model:

| Desktop (`src/`) | Android (`android/.../drowsy/`) |
|---|---|
| `geometry.py` | `Geometry.kt` |
| `drowsiness.py` + `config.py` | `Drowsiness.kt` |
| `preprocess.py` + `model.py` | `EyeClassifier.kt` |
| `alarm.py` | `Alarm.kt` |
| `infer.py` | `MainActivity.kt` + `OverlayView.kt` |

The landmark indices are copied verbatim, and that is safe: they are fixed by
MediaPipe itself, so point 33 is the outer corner of the right eye on every
platform.

### The model

`eyenet.onnx` is the same 139,426-parameter network, exported from the same
checkpoint. It was not assumed to be equivalent — both were run over all
10,076 test images:

```
max probability difference : 4.768e-07
decisions agreeing         : 10,076/10,076
accuracy pytorch / onnx    : 98.65% / 98.65%
```

`scripts/export_onnx.py` exits non-zero on any decision mismatch, because an
export can succeed and still be subtly wrong — an unsupported op quietly
approximated — and the phone would then be worse than the desktop with nothing
to indicate it.

### Why OpenCV is a dependency (23 MB of the APK)

`preprocess.py` applies CLAHE to the eye crop before the CNN sees it. Training
and serving must preprocess **identically** or accuracy collapses silently —
the model keeps returning confident numbers, they are just wrong.

Hand-writing CLAHE in Kotlin would mean matching histogram binning, clip
redistribution and bilinear tile interpolation exactly. Using the same OpenCV
implementation removes that entire class of bug. 23 MB is a fair price for
"cannot silently disagree with training".

Likewise `INTER_AREA` is not interchangeable with `INTER_LINEAR`: for
downscaling it averages the source pixels where `INTER_LINEAR` samples and
aliases. Training used `INTER_AREA`, so the phone does too.

---

## Differences from the desktop version

| | Desktop | Android |
|---|---|---|
| Lighting normalisation | yes (`lighting.py`) | yes (`Lighting.kt`) |
| Head-pose seeding from the previous frame | yes | yes (`Geometry.kt`) |
| Closure strip on the display | yes (`hud.py`) | yes (`OverlayView.kt`) |
| Saved calibration | `calibration.json` | `SharedPreferences` |
| Debug eye-crop view | `d` key | not present |
| Screenshots / recording | `s` / `--record` | not present |
| Session summary on exit | yes | not present |

Night handling is now ported. `Lighting.kt` is the same adaptive gamma + CLAHE
stage, using the OpenCV already bundled, so the maths is the same
implementation rather than a reimplementation. On desktop it takes night from
0.447 to 0.118 `P(closed)`.

`scripts/check_port_parity.py` compares the Kotlin constants and landmark
indices against the Python mechanically, because two copies of the same logic
drift silently. It has already caught two real cases: a dropped `contrast < 60`
term in the lighting classifier, and a stale threshold default. The lighting
stage's own constants are now named on both sides (`LightCfg` in Kotlin) and
checked too — 52 values in all.

Constants are the mechanical half. The logic half has its own suite:
`app/src/test/.../DrowsinessMonitorTest.kt` runs 22 of the desktop scenarios
against the Kotlin state machine on the plain JVM — no device, no emulator —
because `Drowsiness.kt` has no Android dependency:

```bash
cd android
./gradlew testDebugUnitTest
```

---

## Frame handling

`ImageAnalysis` uses `STRATEGY_KEEP_ONLY_LATEST` — the same decision the
desktop `VideoSource` makes for network streams. Process the newest frame and
drop whatever queued behind it, because for a real-time safety signal **fresh
beats complete**. A drowsiness warning describing a microsleep from ten seconds
ago is worse than useless.

Frames arrive rotated, since the camera sensor is mounted at an angle to the
screen. They are rotated upright before anything else touches them: MediaPipe
does not detect a sideways face at all, so this cannot be deferred.

The frame's buffer is wrapped as an OpenCV `Mat` directly, with the row stride
passed as the Mat's step, so the rotation is the only copy made. The earlier
path went buffer → Bitmap → Mat → crop → rotate, two extra full-frame copies,
and could crash outright: Android does not guarantee the last row of a plane
carries its padding, and `Bitmap.copyPixelsFromBuffer` throws on exactly that.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "App not installed" | wrong ABI | Install `app-universal-debug.apk` |
| Black screen | permission denied | Settings → Apps → Drowsy Driver → Permissions → Camera |
| No face detected | too dark, or too far | Move closer. Night correction is active, but software cannot recover a signal the sensor never captured |
| Says drowsy while awake | calibration missed | Tap **Calibrate**, eyes open, hold 3 s |
| Alarm inaudible | media volume | It plays on the **alarm** stream — raise alarm volume, not media |
| Low FPS / phone hot | thermal throttling | Expected. Every threshold is in seconds, not frames, so the verdict does not change — only smoothness does |

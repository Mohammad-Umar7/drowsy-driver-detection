"""
The tiny test harness shared by the suites in this directory.

Each suite runs standalone (`python -m tests.test_drowsiness`) and prints a
PASS/FAIL line per check so the output reads like a report. The same files
also run under pytest - and that is the reason this helper exists.

THE BUG THIS FIXES: check() used to only print and count. Under pytest a test
function that prints "FAIL" and returns normally is a PASSED test. So
`pytest tests/` reported everything green no matter what the checks said,
which is the worst kind of test: one that cannot fail. Under pytest a failed
check now raises, so it fails the test it belongs to.
"""
import sys
import time

_passed = 0
_failed = 0


def wait_until(predicate, timeout: float = 5.0, step: float = 0.01) -> bool:
    """
    Poll until `predicate()` is true, or give up after `timeout` seconds.

    Threaded code must be waited FOR, not slept AROUND. A fixed
    `time.sleep(0.05)` before checking a background thread's result passes
    on an idle machine and fails when the same suite runs beside an emulator
    and a Gradle build - which is exactly what happened. Polling with a
    generous deadline is deterministic: it finishes early when things are
    quick and still passes when they are slow.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


def check(name: str, condition, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
        return
    _failed += 1
    print(f"  FAIL  {name}   {detail}")
    if "pytest" in sys.modules:
        raise AssertionError(f"{name}: {detail}")


def summary(title: str) -> None:
    """Print the totals and exit non-zero if anything failed."""
    print("\n" + "=" * 60)
    print(f"{title}: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)

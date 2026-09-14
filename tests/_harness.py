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

_passed = 0
_failed = 0


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

# alerter.py
import time
import subprocess

PERSIST_SECONDS  = 5   # bad posture must last this long before alerting
COOLDOWN_SECONDS = 2  # minimum gap between two alerts


class posture_alerter:
    def __init__(self):
        self.bad_since  = None  # timestamp when bad posture started
        self.last_alert = None  # timestamp of last alert

    def update(self, bad_posture: bool):
        """
        Call once per frame with the current posture result.
        Fires a sound alert when conditions are met.
        """
        now = time.time()

        if not bad_posture:
            self.bad_since = None  # reset: posture is fine
            return

        # Bad posture detected — start timer if not already running
        if self.bad_since is None:
            self.bad_since = now

        bad_duration = now - self.bad_since

        # Check if bad long enough AND cooldown has passed
        cooldown_passed = (self.last_alert is None) or (now - self.last_alert > COOLDOWN_SECONDS)

        if bad_duration >= PERSIST_SECONDS and cooldown_passed:
            self._alert()
            self.last_alert = now
            self.bad_since  = now  # reset so it needs another PERSIST_SECONDS to fire again

    def _alert(self):
        subprocess.Popen(["say", "fix your posture"])  # non-blocking

import os
import subprocess
import sys
import threading

STEPS = [
    ("build_dataset.py", "Downloading prices and economic data"),
    ("risk_features.py", "Building the risk signals"),
    ("forecast.py", "Testing the risk forecasts year by year"),
    ("vol_target.py", "Testing the forecast in money terms"),
    ("signals.py", "Building the direction signals"),
    ("walk_forward.py", "Testing direction predictions year by year"),
]


class Updater:
    def __init__(self, folder, steps=STEPS):
        self.folder = folder
        self.steps = steps
        self.step = 0
        self.finished = False
        self.error = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def status(self):
        text = self.steps[min(self.step, len(self.steps) - 1)][1]
        return f"Step {self.step + 1} of {len(self.steps)}: {text}..."

    def log_path(self):
        return os.path.join(self.folder, "data", "update_log.txt")

    def write_log(self, text):
        os.makedirs(os.path.join(self.folder, "data"), exist_ok=True)
        with open(self.log_path(), "w", encoding="utf-8") as file:
            file.write(text)

    def run(self):
        if os.path.exists(self.log_path()):
            os.remove(self.log_path())
        environment = dict(os.environ)
        environment["MPLBACKEND"] = "Agg"
        for index, (script, text) in enumerate(self.steps):
            self.step = index
            try:
                result = subprocess.run(
                    [sys.executable, os.path.join(self.folder, script)],
                    cwd=self.folder,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=environment,
                    check=False,
                )
            except OSError as error:
                self.error = f"{text} failed: {error}"
                break
            if result.returncode != 0:
                lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
                reason = lines[-1] if lines else "unknown error"
                self.error = f"{text} failed.\n\n{reason}"
                self.write_log(result.stdout + "\n" + result.stderr)
                break
        self.finished = True

import os

from updater import Updater


def write_script(folder, name, code):
    with open(os.path.join(folder, name), "w") as file:
        file.write(code)


def test_updater_runs_all_steps(tmp_path):
    write_script(tmp_path, "one.py", "open('one.txt', 'w').write('x')")
    write_script(tmp_path, "two.py", "open('two.txt', 'w').write('y')")
    updater = Updater(str(tmp_path), steps=[("one.py", "First"), ("two.py", "Second")])
    updater.start()
    updater.thread.join(timeout=60)
    assert updater.finished
    assert updater.error is None
    assert (tmp_path / "one.txt").exists()
    assert (tmp_path / "two.txt").exists()


def test_updater_stops_on_failure_and_writes_log(tmp_path):
    write_script(tmp_path, "bad.py", "import sys\nprint('partial output')\nsys.exit('Could not download prices')")
    write_script(tmp_path, "after.py", "open('after.txt', 'w').write('z')")
    updater = Updater(str(tmp_path), steps=[("bad.py", "Downloading"), ("after.py", "Later")])
    updater.start()
    updater.thread.join(timeout=60)
    assert updater.finished
    assert "Could not download prices" in updater.error
    assert not (tmp_path / "after.txt").exists()
    assert "partial output" in (tmp_path / "data" / "update_log.txt").read_text()


def test_updater_status_names_the_step(tmp_path):
    updater = Updater(str(tmp_path), steps=[("a.py", "First"), ("b.py", "Second")])
    assert updater.status() == "Step 1 of 2: First..."
    updater.step = 1
    assert updater.status() == "Step 2 of 2: Second..."

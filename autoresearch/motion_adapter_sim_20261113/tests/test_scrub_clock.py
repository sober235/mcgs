"""The job folder name embeds the server's clock; nothing that is committed may carry it."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import scrub_clock  # noqa: E402


def test_the_clock_part_of_a_job_folder_is_removed_and_the_suffix_kept():
    s = "/home/u/.cache/mcgs-vscode/jobs/j-20200102T030405-ab12cd34/code/x.py"
    assert scrub_clock.scrub(s) == "/home/u/.cache/mcgs-vscode/jobs/j-ab12cd34/code/x.py"


def test_text_without_a_job_folder_is_unchanged():
    s = '{"local_start_pst": "2026-11-22 08:22:10 PST", "job": "ab12cd34"}'
    assert scrub_clock.scrub(s) == s


def test_a_file_is_rewritten_once_and_then_left_alone(tmp_path):
    p = tmp_path / "experiments.jsonl"
    rows = [dict(exp="a", argv=["/jobs/j-20200102T030405-ab12cd34/code/x.py", "--exp", "a"]), dict(exp="b", argv=[])]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert scrub_clock.scrub_file(p) == 1                    # one line changed
    out = [json.loads(line) for line in p.read_text().splitlines()]
    assert out[0]["argv"][0] == "/jobs/j-ab12cd34/code/x.py" and out[1] == rows[1]
    assert scrub_clock.scrub_file(p) == 0

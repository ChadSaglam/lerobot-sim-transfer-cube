import json
from pathlib import Path

import collect_results


def _write_eval(root: Path, job: str, n: int, n_success: int, ci: tuple[float, float]) -> None:
    d = root / "outputs" / "eval" / job
    d.mkdir(parents=True)
    info = {
        "overall": {
            "pc_success": 100.0 * n_success / n,
            "n_episodes": n,
            "n_success": n_success,
            "pc_success_ci95": list(ci),
            "avg_max_reward": 3.2,
        }
    }
    (d / "eval_info.json").write_text(json.dumps(info))


def test_table_sorted_and_formatted(tmp_path, monkeypatch):
    _write_eval(tmp_path, "act_seed1000_050000_n50", 50, 31, (48.2, 74.1))
    _write_eval(tmp_path, "act_seed1000_010000_n50", 50, 33, (52.2, 77.6))
    _write_eval(tmp_path, "baseline_official_act_n200", 200, 162, (75.0, 85.8))
    monkeypatch.chdir(tmp_path)

    collect_results.main()

    md = (tmp_path / "results" / "results.md").read_text().splitlines()
    assert md[0].startswith("| Model | Steps | Episodes |")
    rows = md[2:]
    assert rows[0].startswith(
        "| official lerobot/act_aloha_sim_transfer_cube_human | 80k (reported) | 200 "
        "| 81.0% (162/200) | 75.0-85.8 |"
    )
    assert rows[1].startswith("| own ACT, seed 1000 | 10k | 50 | 66.0% (33/50) | 52.2-77.6 |")
    assert rows[2].startswith("| own ACT, seed 1000 | 50k | 50 | 62.0% (31/50) | 48.2-74.1 |")

    csv_lines = (tmp_path / "results" / "results.csv").read_text().splitlines()
    assert csv_lines[0] == "job,model,step,n_episodes,n_success,success_pct,ci95_low,ci95_high,avg_max_reward"
    assert len(csv_lines) == 4


def test_describe_unknown_job_is_kept():
    assert collect_results.describe("something_else") == ("something_else", "-", 10**9)

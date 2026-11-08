from pathlib import Path

import plot_training_curves as ptc

# One physical line of a lerobot-train log after replacing "\r" with "\n":
# the tqdm counter and the INFO metrics line share it.
SAMPLE = (
    "Training:  27%|██▋       | 13400/50000 [1:37:42<4:36:02,  2.21step/s]"
    "INFO 2026-11-07 20:18:58 ot_train.py:894 step:13K smpl:107K ep:268 epch:5.36 "
    "loss:0.140 grdn:12.544 lr:1.0e-05 data_s:0.003 prep_s:0.000 updt_s:0.436\n"
)


def test_read_loss_uses_exact_tqdm_step(tmp_path):
    log = tmp_path / "act_transfer_cube_seed1000.log"
    second = (
        "Training: 28%| 13600/50000 [1:39:13<4:27:21, 2.27step/s]"
        "INFO 2026-11-07 20:20:29 ot_train.py:894 step:14K smpl:109K loss:0.137 grdn:12.2 lr:1.0e-05 \n"
    )
    log.write_text(SAMPLE.replace("\n", "\r") + second)
    assert ptc.read_loss(log) == [(13400, 0.140), (13600, 0.137)]


def test_plots_are_written(tmp_path, monkeypatch):
    train = tmp_path / "outputs" / "train"
    train.mkdir(parents=True)
    (train / "act_transfer_cube_seed1000.log").write_text(SAMPLE)
    results = tmp_path / "results"
    results.mkdir()
    (results / "results.csv").write_text(
        "job,model,step,n_episodes,n_success,success_pct,ci95_low,ci95_high,avg_max_reward\n"
        "baseline_official_act_n200,official x,80k (reported),200,162,81.0,75.0,85.8,3.5\n"
        "act_seed1000_010000_n50,\"own ACT, seed 1000\",10k,50,33,66.0,52.2,77.6,3.2\n"
        "act_seed1000_050000_n200,\"own ACT, seed 1000\",50k,200,128,64.0,57.1,70.3,3.1\n"
    )
    monkeypatch.chdir(tmp_path)

    ptc.main()

    assert (results / "training_loss.png").stat().st_size > 0
    assert (results / "success_vs_steps.png").stat().st_size > 0


def test_fixture_path_exists():
    # keep the test file location stable for the sys.path shim in conftest.py
    assert (Path(__file__).parent / "conftest.py").exists()

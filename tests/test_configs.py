from rl_dev.constants import MANISKILL_SAC_DIR
from rl_dev.utils.config import load_launcher_config


def test_state_config_loads():
    config = load_launcher_config("configs/phase1_lift_state.yaml")
    assert config["args"]["env_id"] == "PandaLiftCube-v0"


def test_rgb_config_contains_oom_ladder():
    config = load_launcher_config("configs/phase2_lift_rgb.yaml")
    assert config["args"]["buffer_size"] == 150000
    assert len(config["oom_fallback_ladder"]) == 3


def test_ddpg_pickcube_rgb_config_contains_state_and_oom_ladder():
    config = load_launcher_config("configs/pickcube_rgb_ddpg.yaml")
    assert config["args"]["env_id"] == "PickCube-v1"
    assert config["args"]["include_state"] is True
    assert len(config["oom_fallback_ladder"]) == 3


def test_maniskill_baseline_source_exists():
    assert (MANISKILL_SAC_DIR / "sac.py").exists()
    assert (MANISKILL_SAC_DIR / "sac_rgbd.py").exists()

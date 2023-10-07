from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
MANISKILL_TAG = "v3.0.0b22"
MANISKILL_COMMIT = "33967b9e3ead1f841eec57cc9f31d0d8b8cf0907"
MANISKILL_SOURCE_ROOT = THIRD_PARTY_ROOT / "ManiSkill"
MANISKILL_SAC_DIR = MANISKILL_SOURCE_ROOT / "examples" / "baselines" / "sac"
DEFAULT_ASSET_DIR = Path.home() / ".maniskill" / "data"


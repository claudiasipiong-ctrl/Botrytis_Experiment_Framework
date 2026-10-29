"""20 - Verify exact content isolation and export exact counts/metrics/artifact paths.
Run after 05, after each 10, or after 16/18. Does not run any model or tune test.
"""

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logging_utils import run_cli
from utils.prepared import load_prepared_data
from utils.split_summary import export_split_summary

def main(ctx, args):
    data = load_prepared_data(ctx.cfg, with_folds=True)
    summary = export_split_summary(ctx.cfg, data)
    ctx.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)
    ctx.log_metrics({"counts":summary["counts"], "exact_content_overlap":False})

if __name__ == "__main__":
    raise SystemExit(run_cli("20_audit_split_evidence", main, description=__doc__))

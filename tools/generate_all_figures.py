"""Regenerate figures from verified CSV descriptors; pending is a valid faithful state."""
from __future__ import annotations

import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root=_BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0,str(_boot_root/"src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__,_boot_root)
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from utils.figures_all import generate
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-index',type=Path);p.add_argument('--output',type=Path,default=ROOT/'reports/v4/all_figures')
    p.add_argument('--catalog-only',action='store_true');p.add_argument('--allow-pixels',action='store_true');p.add_argument('--allow-final-test',action='store_true');p.add_argument('--config',default='configs/local.yaml')
    a=p.parse_args();index=json.loads(a.source_index.read_text(encoding='utf-8')) if a.source_index else {}
    if a.allow_final_test:
        from utils.config_loader import load_config
        from utils.splits import assert_split_allowed
        assert_split_allowed('test','final_test',load_config(a.config))
    rows=generate(a.output,index,a.catalog_only,a.allow_pixels,a.allow_final_test)
    from collections import Counter
    print(json.dumps(dict(Counter(r['status'] for r in rows)),ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())

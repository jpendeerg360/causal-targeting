"""Repository CLI. Run from the repository root or pass --root."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'prepare', 'develop', 'evaluate', 'report', 'verify', 'run'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--data-file', type=Path, help='Existing publisher v2.1 .csv.gz; checksum still enforced')
    parser.add_argument('--rebuild', action='store_true', help='Archive prior aggregate results locally, then repeat the fixed analysis')
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'config/analysis.json').exists():
        parser.error('Run from repository root or set --root to it')
    config = json.loads((root / 'config/analysis.json').read_text())
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(config['threads']))
    os.environ.setdefault('OMP_NUM_THREADS', str(config['threads']))
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'artifacts/matplotlib'))
    for relative in ['artifacts', 'results/tables', 'results/figures', 'data/raw', 'data/processed']:
        (root / relative).mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        if args.command not in ['run', 'develop']:
            parser.error('--rebuild applies only to run or develop')
        backup = root / 'artifacts' / ('prior_results_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        shutil.copytree(root / 'results', backup)
        (root / 'results/freeze.json').unlink(missing_ok=True)
    from .data import download, prepare
    if args.command == 'download':
        print(download(root, args.data_file))
    elif args.command == 'prepare':
        prepare(root, config, args.data_file)
    elif args.command == 'develop':
        from .pipeline import develop
        develop(root, config)
    elif args.command == 'evaluate':
        from .pipeline import evaluate
        evaluate(root, config)
    elif args.command == 'report':
        from .reporting import report
        report(root, config)
    elif args.command == 'verify':
        from .verification import verify
        verify(root, config)
    elif args.command == 'run':
        from .pipeline import develop, evaluate
        from .reporting import report
        from .verification import verify
        prepare(root, config, args.data_file)
        if (root / 'results/freeze.json').exists():
            if not (root / 'artifacts/tree.joblib').exists():
                parser.error('Published results are present but models are absent; use run --rebuild to reproduce')
        else:
            develop(root, config)
        evaluate(root, config)
        report(root, config)
        verify(root, config)


if __name__ == '__main__':
    main()

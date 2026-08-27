#!/usr/bin/env python
"""Generate bundle READMEs from emulator and optional validation metadata."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))

from hi_fast import HiFast  # noqa: E402


def _available_models(root):
    """Return emulator bundle names under ``root``."""
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def main():
    parser = argparse.ArgumentParser(
        description='Generate README.md files for HiFast emulator bundles.')
    parser.add_argument(
        '--root',
        default=str(REPO_ROOT / 'emu'),
        help='Directory containing emulator bundle folders. Default: emu')
    parser.add_argument(
        '--model',
        action='append',
        help=('Bundle name to document. Can be passed multiple times. '
              'Default: every subfolder under --root.'))
    parser.add_argument(
        '--bounds',
        choices=('thin', 'std', 'ext'),
        default=None,
        help='Optional trust region to document instead of all regions.')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise OSError('Emulator root {} does not exist'.format(root))

    models = args.model or _available_models(root)
    for model in models:
        output = root / model / 'README.md'
        print('Generating {}'.format(output))
        hifast = HiFast(model, root=str(root))
        hifast.print_info(
            bounds=args.bounds,
            markdown=True,
            output=str(output))


if __name__ == '__main__':
    main()

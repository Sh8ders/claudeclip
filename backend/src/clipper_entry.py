"""Console entry point for ``clipper`` (Claude-native mode).

The editable install exposes ``backend/src`` as the import root, while the
application imports itself as the ``src`` package. Put ``backend/`` on the
path first so ``src.native.cli`` and its relative imports resolve.
"""

import sys
from pathlib import Path


def main() -> int:
    backend = str(Path(__file__).resolve().parents[1])
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from src.native.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())

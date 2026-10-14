"""Generate and save new evaluation environments with the library's own generator.

    python -m lpac_study.make_environments --data-dir /path/to/lpac --out /path/to/lpac_new --count 40

Each environment is drawn by ``CoverageSystem(params)`` exactly as the authors' evaluator
does when an environment file is missing (32 Gaussian features, uniform initial
positions), then written to ``<out>/envs/<i>.env`` and ``<out>/envs/<i>.pos`` so that every
later run reuses the same files.  ``<out>/models`` and ``<out>/params`` are symbolic links to
the release folder, which makes ``<out>`` a drop-in ``--data-dir`` for the other scripts.
The simulator seeds its generator from ``std::random_device``: the saved files, not a seed,
are the reference.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import coverage_control as cc
from coverage_control import CoverageSystem

from .paths import resolve_data_dir


def generate(out: Path, env_params_file: Path, count: int) -> int:
    """Write ``count`` new environments; existing files are kept (never overwritten)."""
    envs = out / "envs"
    envs.mkdir(parents=True, exist_ok=True)
    params = cc.Parameters(str(env_params_file))
    written = 0
    for index in range(count):
        env_file, pos_file = envs / f"{index}.env", envs / f"{index}.pos"
        if env_file.is_file() and pos_file.is_file():
            continue
        env = CoverageSystem(params)
        env.WriteEnvironment(str(pos_file), str(env_file))
        written += 1
    return written


def link_release(out: Path, release_root: Path) -> None:
    """Expose the release's models and parameters next to the new environments."""
    for name in ("models", "params"):
        target = out / name
        if not target.exists():
            os.symlink(release_root / name, target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", help="release folder (or $LPAC_DATA_DIR)")
    parser.add_argument("--out", required=True, help="folder receiving envs/ plus links to models/ and params/")
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    paths = resolve_data_dir(args.data_dir)
    out = Path(args.out).expanduser().resolve()
    written = generate(out, paths.env_params, args.count)
    link_release(out, paths.root)
    print(f"{written} new environments written, {args.count} available in {out / 'envs'}")


if __name__ == "__main__":
    main()

"""Run the unmodified official checker with a POSIX map path on Windows hosts.

Usage: python tools/check_docker.py --map-dir /arena -- docker run ...
Only the checker's abspath('/arena') is normalized; transport, schema and timing
checks are the official kit implementation.
"""
import os
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[1]
original_abspath = os.path.abspath
if os.name == 'nt':
    os.path.abspath = lambda path: '/arena' if os.fspath(path) == '/arena' else original_abspath(path)
runpy.run_path(str(root/'mechfight-kit/check/check.py'), run_name='__main__')

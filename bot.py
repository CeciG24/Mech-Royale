"""Entry point: the passive survival agent behind the kit's stdin/stdout wrapper."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mechfight as mf  # noqa: E402
from brain.agent import Agent  # noqa: E402

if __name__ == "__main__":
    mf.run(Agent)

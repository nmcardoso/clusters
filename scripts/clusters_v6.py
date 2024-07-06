import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline


def clusters_v6_pipeline():
  pass


if __name__ == "__main__":
  clusters_v6_pipeline()
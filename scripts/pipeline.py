import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

from splusclusters._pipelines import all_clusters_pipeline
from splusclusters.configs import configs


def main(args):
  configs.VERSION = args.version
  all_clusters_pipeline(
    version=args.version,
    skip_cones=args.skip_cones,
    skip_plots=args.skip_plots,
    skip_website=args.skip_website,
    overwrite=args.overwrite,
    photoz_odds=args.photoz_odds,
    separated=args.separated,
    splus_only=args.splus_only,
    fmt=args.fmt,
    two=args.two,
  )


if __name__ == '__main__':
  parser = ArgumentParser('python pipeline.py', description='main SCALE pipeline')
  parser.add_argument('--version', '-v', action='store', type=int, choices=[5, 6, 7], default=7, help='catalog version, default: 7')
  parser.add_argument('--skip-cones', action='store_true')
  parser.add_argument('--skip-plots', action='store_true')
  parser.add_argument('--skip-website', action='store_true')
  parser.add_argument('--separated', action='store_true', help='separated plots')
  parser.add_argument('--splus-only', action='store_true', help='only render plots with splus data')
  parser.add_argument('--two', action='store_true', help='process only A168 and MKW4')
  parser.add_argument('--odds', action='store', type=float, default=0.9, help='photoz odds, default: 0.9')
  parser.add_argument('--fmt', action='store', choices=['pdf', 'png'], default='png', help='plots format, default: png')
  args = parser.parse_args()
  main(args)
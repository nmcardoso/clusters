import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline

from splusclusters.constants import *
from splusclusters.loaders import (LoadAllRadialStage, LoadClusterInfoStage,
                                   load_clusters)
from splusclusters.plots import MagDiffPlotStage


def magdiff_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  # df_erass = load_eRASS()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadAllRadialStage(),
    MagDiffPlotStage(overwrite=overwrite),
  )
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  # pipe = Pipeline(
  #   LoadERASSInfoStage(df_erass),
  #   LoadAllRadialStage(),
  #   MagDiffPlotStage(overwrite=overwrite),
  # )
  # pipe.map_run('cls_name', df_erass.Cluster.values, workers=1)
  
  plot_paths = sorted(MAGDIFF_PLOTS_FOLDER.glob('cls_*.pdf'))
  concat_plot_path = MAGDIFF_PLOTS_FOLDER / 'magdiff_v2.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  
if __name__ == "__main__":
  magdiff_pipeline()
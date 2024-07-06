import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline

from splusclusters.constants import *
from splusclusters.loaders import LoadClusterInfoStage, load_clusters
from splusclusters.plots import VelocityPlotStage


def velocity_plots_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    VelocityPlotStage(overwrite=overwrite)
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  plot_paths = sorted(VELOCITY_PLOTS_FOLDER.glob('cls_*.pdf'))
  concat_plot_path = VELOCITY_PLOTS_FOLDER / 'velocity_plots_v1.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  

if __name__ == "__main__":
  velocity_plots_pipeline()
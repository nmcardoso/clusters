import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.loaders import (LoadClusterInfoStage, load_clusters,
                                   load_photoz, load_spec)
from splusclusters.match import PhotoZRadialSearchStage, SpecZRadialSearchStage
from splusclusters.plots import ClusterPlotStage


def spec_plots_pipeline():
  df_clusters = load_clusters()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=True),
    SpecZRadialSearchStage(overwrite=False),
    ClusterPlotStage(overwrite=True)
  )
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  df2 = df_clusters.sort_values('ra')
  plot_paths = [PLOTS_FOLDER / f'cls_{c}.pdf' for c in df2.name.values]
  concat_plot_path = PLOTS_FOLDER / 'clusters_v6_RA.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  

if __name__ == '__main__':
  spec_plots_pipeline()
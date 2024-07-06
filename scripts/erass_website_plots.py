
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.loaders import (LoadERASSInfoStage, LoadLegacyRadialStage,
                                   load_eRASS, load_photoz, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 SpecZRadialSearchStage, StarsRemovalStage)
from splusclusters.plots import ClusterPlotStage


def erass_website_plots_pipeline():
  # df_clusters = load_full_eRASS()
  df_clusters = load_eRASS()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    LoadLegacyRadialStage(),
    StarsRemovalStage(),
    ClusterPlotStage(overwrite=False, fmt='jpg', output_folder='outputs_v6/website_plots')
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1, validate=False)
  

if __name__ == '__main__':
  erass_website_plots_pipeline()
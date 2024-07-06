
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.loaders import (LoadClusterInfoStage, LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_clusters, load_photoz)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def match_all_pipeline():
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=False),
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
if __name__ == "__main__":
  match_all_pipeline()
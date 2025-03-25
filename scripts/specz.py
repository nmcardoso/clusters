import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.loaders import (LoadClusterInfoStage, load_clusters,
                                   load_spec)
from splusclusters.match import SpecZRadialSearchStage
from splusclusters.plots import ClusterPlotStage


def spec_pipeline(overwrite: bool = False):
  df_clusters = load_catalog_v6()
  df_spec, specz_skycoord = load_spec()
  
  configs.Z_SPEC_DELTA = 0.02
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    SpecZRadialSearchStage(overwrite=overwrite),
  )
  
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  
if __name__ == '__main__':
  spec_pipeline()
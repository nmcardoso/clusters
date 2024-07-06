import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.loaders import LoadERASSInfoStage, load_eRASS, load_spec
from splusclusters.match import SpecZRadialSearchStage


def spec_erass_pipeline(overwrite: bool = False):
  df_clusters = load_eRASS()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    SpecZRadialSearchStage(overwrite=overwrite)
  )
  
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)
  

if __name__ == "__main__":
  spec_erass_pipeline()
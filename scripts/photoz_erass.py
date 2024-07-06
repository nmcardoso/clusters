import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.loaders import LoadERASSInfoStage, load_eRASS, load_photoz
from splusclusters.match import PhotoZRadialSearchStage


def photoz_erass_pipeline(overwrite: bool = False):
  df_clusters = load_eRASS()
  df_photoz, photoz_skycoord = load_photoz()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=overwrite),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)
  

if __name__ == "__main__":
  photoz_erass_pipeline()
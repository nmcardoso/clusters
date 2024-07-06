import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline

from splusclusters.loaders import (LoadAllRadialStage, LoadClusterInfoStage,
                                   load_clusters)


def magdiff_outliers_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadAllRadialStage(),
    MagdiffOutlierStage(overwrite),
  )
  
  pipe.map_run('cls_id', [12, 27], workers=1)
  

if __name__ == "__main__":
  magdiff_outliers_pipeline()
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline

from splusclusters.external import CopyXrayStage
from splusclusters.loaders import (LoadAllRadialStage, LoadClusterInfoStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_clusters)
from splusclusters.plots import (ClusterPlotStage, MagDiffPlotStage,
                                 VelocityPlotStage)
from splusclusters.website import WebsitePagesStage


def website_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=overwrite, fmt='jpg', separated=True),
    VelocityPlotStage(overwrite=overwrite, fmt='jpg', separated=True),
    MagDiffPlotStage(overwrite=overwrite, fmt='jpg', separated=True),
    CopyXrayStage(overwrite=overwrite, fmt='png'),
    WebsitePagesStage(clusters=df_clusters.name.values),
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  WebsitePagesStage(clusters=df_clusters.name.values).make_index()
  

if __name__ == '__main__':
  website_pipeline()
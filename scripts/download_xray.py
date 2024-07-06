
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline

from splusclusters.loaders import LoadClusterInfoStage, load_clusters
from splusclusters.xray import DownloadXRayStage


def download_xray_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    DownloadXRayStage(overwrite=overwrite, fmt='png')
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  

if __name__ == "__main__":
  download_xray_pipeline()
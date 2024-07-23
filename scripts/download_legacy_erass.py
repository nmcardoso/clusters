import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline

from splusclusters.configs import configs
from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import LoadERASSInfoStage, load_eRASS


def download_legacy_erass_pipeline(clear: bool = False):
  df_clusters = load_eRASS()
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    DownloadLegacyCatalogStage('cls_search_radius_deg', overwrite=False, workers=8)
  )
  ls10_pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)
  

if __name__ == "__main__":
  download_legacy_erass_pipeline()
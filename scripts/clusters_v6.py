import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import (LoadAllRadialStage, LoadLegacyRadialStage,
                                   LoadPauloInfoStage, LoadPhotozRadialStage,
                                   LoadSpeczRadialStage, load_catalog_v6,
                                   load_photoz2, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def clusters_v6_pipeline(clear: bool = False):
  df_clusters = load_catalog_v6()
  df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SPECZ_FOLDER = configs.OUT_PATH / 'specz_paulo'
  configs.setup_paths()
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  print(df_clusters)
        
  ls10_pipe = Pipeline(
    LoadPauloInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=False, workers=8),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=False),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=False, splus_only=False),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)


if __name__ == "__main__":
  clusters_v6_pipeline()
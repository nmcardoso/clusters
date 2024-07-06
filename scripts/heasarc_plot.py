
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import (LoadAllRadialStage, LoadHeasarcInfoStage,
                                   LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_heasarc, load_photoz, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def heasarc_plot_pipeline(overwrite: bool = False):
  df_heasarc = load_heasarc()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadHeasarcInfoStage(df_heasarc),
    PhotoZRadialSearchStage(overwrite=overwrite),
    SpecZRadialSearchStage(overwrite=overwrite),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=overwrite),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(),
    LoadAllRadialStage(),
    ClusterPlotStage(),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', ['[YMV2007]1854', '[YMV2007]337642', 'ACT-CLJ0006.9-0041', '[YMV2007]15744', '[YMV2007]337638',], workers=1)
  
  
if __name__ == "__main__":
  heasarc_plot_pipeline()
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astropy.wcs import WCS
from matplotlib import pyplot as plt

from splusclusters.loaders import LoadClusterInfoStage, load_clusters


def all_sky_plot():
  df_clusters = load_clusters()
  df_clusters = df_clusters.sort_values('name').reset_index()
  info = LoadClusterInfoStage(df_clusters)
  
  wcs_spec =  {
    'CRVAL1': 180,
    'CRVAL2': 0,
    'CTYPE1': 'RA---AIT',
    'CTYPE2': 'DEC--AIT',
    'CUNIT1': 'deg',
    'CUNIT2': 'deg'
  }
  wcs = WCS(wcs_spec)
  fig = plt.figure(figsize=(18.5, 8), dpi=150)
  ax = fig.add_subplot(projection=wcs)
  cmap = plt.cm.get_cmap('prism', len(df_clusters))
    
  for i, row in df_clusters.iterrows():
    cluster = info.run(row['clsid'])
    df_members = cluster['df_members']
    cls_name = cluster['cls_name']
    ax.scatter(
      df_members.ra, 
      df_members.dec, 
      label=cls_name,
      color=cmap(i), 
      s=1,
      transform=ax.get_transform('icrs'),
      rasterized=True,
    )
  ax.grid('on', color='k', linestyle='--', alpha=.5)
  ax.tick_params(direction='in')
  ax.set_xlabel('RA')
  ax.set_ylabel('DEC')
  ax.set_title('S-PLUS Clusters Catalog')
  box = ax.get_position()
  # ax.set_position([box.x0, box.y0 + box.height * 0.3,
  #                box.width, box.height * 0.7])
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
           fancybox=True, shadow=False, ncol=10)
  plt.savefig('docs/all_sky.png', bbox_inches='tight', pad_inches=0.1)
  plt.close(fig)
from typing import List, Literal, Sequence, Tuple

import astropy.units as u
import dagster as dg
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import seaborn as sns
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

from splusclusters._loaders import ClusterInfo
from splusclusters.configs import configs
from splusclusters.utils import cond_overwrite


def _get_plot_title(info: ClusterInfo):
  return (
    f'Cluster: {info.name} (RA: {info.ra:.5f}, DEC: {info.dec:.5f})\n'
    f'Search Radius: {info.search_radius_Mpc:.2f}Mpc = {info.search_radius_deg:.3f}$^\\circ$ ($z_{{cluster}}={info.z:.4f}$)\n'
    f'$z_{{spec}}$: $z_{{cluster}} \\pm {info.z_spec_delta}$ = [{info.z_spec_range[0]:.4f}, {info.z_spec_range[1]:.4f}]\n'
    f'$z_{{photo}}$: $z_{{cluster}} \\pm {info.z_photo_delta}$ = [{info.z_photo_range[0]:.4f}, {info.z_photo_range[1]:.4f}]\n'
    f'R Mag Range: [13, 22] $\\cdot$ Spec Class = GALAXY*\n'
  )


def _add_circle(
  ra: float, 
  dec: float, 
  radius: float,
  color: str,
  ax: plt.Axes,
  label: str = '',
  ls: str = '-',
): 
  circle = SphericalCircle(
    center=[ra, dec]*u.deg,
    radius=radius*u.deg,
    fc='none', 
    lw=2, 
    linestyle=ls,
    ec=color, 
    transform=ax.get_transform('icrs'), 
    label=label,
  )
  ax.add_patch(circle)


def _add_all_circles(info: ClusterInfo, ax: plt.Axes):
  if info.r200_deg:
    _add_circle(
      ra=info.ra,
      dec=info.dec,
      radius=5*info.r200_deg,
      color='tab:green',
      label=f'5$\\times$R200 ({5*info.r200_Mpc:.2f}Mpc $\\bullet$ {5*info.r200_deg:.2f}$^\\circ$)',
      ax=ax
    )
  if info.r500_deg:
    _add_circle(
      ra=info.ra,
      dec=info.dec,
      radius=5*info.r500_deg,
      color='tab:green',
      ls='--',
      label=f'5$\\times$R500 ({5*info.r500_Mpc:.2f}Mpc $\\bullet$ {5*info.r500_deg:.2f}$^\\circ$)',
      ax=ax
    )
  if info.search_radius_deg:
    _add_circle(
      ra=info.ra,
      dec=info.dec,
      radius=info.search_radius_deg,
      color='tab:brown',
      label=f'{info.search_radius_Mpc:.2f}Mpc ({info.search_radius_deg:.3f}$^\\circ$)',
      ax=ax
    )
  
  
def _add_cluster_center(info: ClusterInfo, ax: plt.Axes):
  ax.scatter(
    info.ra, 
    info.dec, 
    marker='+', 
    linewidths=2, 
    s=140, 
    c='tab:red', 
    rasterized=True, 
    transform=ax.get_transform('icrs'),
  )


def _plot_specz(
  info: ClusterInfo,
  df_specz_radial: pd.DataFrame,
  df_members: pd.DataFrame | None,
  df_interlopers: pd.DataFrame | None,
  ax: plt.Axes,
):
  df_plot = df_specz_radial[df_specz_radial.z.between(*info.z_spec_range)]
  ra_col, dec_col = guess_coords_columns(df_plot)
  if df_members is not None and df_interlopers is not None:
    ax.scatter(
      df_plot[ra_col].values, 
      df_plot[dec_col].values, 
      c='tab:red', 
      s=6, 
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label=f'Unclassified'
    )
    ra_col, dec_col = guess_coords_columns(df_members)
    ax.scatter(
      df_members[ra_col].values, 
      df_members[dec_col].values, 
      c='tab:blue', 
      s=6, 
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label=f'Member ({len(df_members)})'
    )
    ax.scatter(
      df_interlopers[ra_col].values, 
      df_interlopers[dec_col].values, 
      c='tab:orange', 
      s=6, 
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label=f'Interloper ({len(df_interlopers)})'
    )
  else:
    ax.scatter(
      df_plot[ra_col].values, 
      df_plot[dec_col].values, 
      c='tab:blue', 
      s=6, 
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label=f'$z_{{spec}}$'
    )
  _add_cluster_center(info, ax)
  _add_all_circles(info, ax)
  ax.set_title(f'$z_{{spec}}$ - Objects: {len(df_plot)}')
  ax.invert_xaxis()
  ax.legend(loc='upper left')
  ax.set_aspect('equal')
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('RA')
  ax.set_ylabel('DEC')



def _plot_photoz(
  info: ClusterInfo, 
  df_photoz_radial: pd.DataFrame, 
  ax: plt.Axes
):
  if df_photoz_radial is not None and len(df_photoz_radial) > 0:
    ra_col, dec_col = guess_coords_columns(df_photoz_radial)
    ax.scatter(
      df_photoz_radial[ra_col].values, 
      df_photoz_radial[dec_col].values,
      c='tab:blue', 
      s=2, 
      alpha=0.001 if len(df_photoz_radial) > 1_000_000 else 0.1,
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label=f'iDR5 objects'
    )
  _add_cluster_center(info, ax)
  _add_all_circles(info, ax)
  ax.set_title(f'S-PLUS Coverage - Objects: {len(df_photoz_radial)}')
  ax.invert_xaxis()
  ax.legend(loc='upper left')
  ax.set_aspect('equal')
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('RA')
  ax.set_ylabel('DEC')
    
def plot_legacy_coverage(
  info: ClusterInfo,
  df_legacy_radial: pd.DataFrame,
  ax: plt.Axes,
):
  if df_legacy_radial is not None and len(df_legacy_radial) > 0:
    ra_col, dec_col = guess_coords_columns(df_legacy_radial)
    df = df_legacy_radial#[df_legacy_radial.type != 'PSF']
    if len(df) > 0:
      ax.scatter(
        df[ra_col].values, 
        df[dec_col].values,
        c='tab:blue', 
        s=2, 
        alpha=0.05 if len(df) > 1_000_000 else 0.1,
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Legacy objects'
      )
      ax.set_title(f'Legacy Survey Coverage - Objects: {len(df)}')
  else:
    ax.set_title(f'Legacy Survey Coverage - Objects: 0')
  _add_cluster_center(info, ax)
  _add_all_circles(info, ax)
  ax.invert_xaxis()
  ax.legend(loc='upper left')
  ax.set_aspect('equal')
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('RA')
  ax.set_ylabel('DEC')




def _plot_photoz_specz(
  info: ClusterInfo,
  df_specz_radial: pd.DataFrame,
  df_photoz_radial: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  photoz_odds: float,
  ax: plt.Axes,
):
  if len(df_specz_radial) > 0 and len(df_photoz_radial) > 0:
    df_photoz_good = df_all_radial[df_all_radial.zml.between(*info.z_photo_range) & (df_all_radial.odds > photoz_odds)]
    df_photoz_good_with_spec = df_photoz_good[df_photoz_good.z.between(*info.z_spec_range)]
    df_photoz_good_wo_spec = df_photoz_good[~df_photoz_good.z.between(*info.z_spec_range) | df_photoz_good.z.isna()]
    df_photoz_bad = df_all_radial[~df_all_radial.zml.between(*info.z_photo_range) & (df_all_radial.odds > photoz_odds)]
    df_photoz_bad_with_spec = df_photoz_bad[df_photoz_bad.z.between(*info.z_spec_range)]
    if len(df_photoz_good_wo_spec) > 0:
      ra_col, dec_col = guess_coords_columns(df_photoz_good_wo_spec)
      ax.scatter(
        df_photoz_good_wo_spec[ra_col].values, 
        df_photoz_good_wo_spec[dec_col].values, 
        c='tab:olive', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'good $z_{{photo}}$ wo/ $z_{{spec}}$ ({len(df_photoz_good_wo_spec)} obj)'
      )
    if len(df_photoz_bad_with_spec) > 0:
      ra_col, dec_col = guess_coords_columns(df_photoz_bad_with_spec)
      ax.scatter(
        df_photoz_bad_with_spec[ra_col].values, 
        df_photoz_bad_with_spec[dec_col].values, 
        c='tab:orange', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'False Negatives ({len(df_photoz_bad_with_spec)} obj)'
      )
    if len(df_photoz_good_with_spec) > 0:
      ra_col, dec_col = guess_coords_columns(df_photoz_good_with_spec)
      ax.scatter(
        df_photoz_good_with_spec[ra_col].values, 
        df_photoz_good_with_spec[dec_col].values, 
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'True Positives ({len(df_photoz_good_with_spec)} obj)'
      )
  _add_cluster_center(info, ax)
  _add_all_circles(info, ax)
  ax.set_title(f'$z_{{photo}}$ $\\cap$ $z_{{spec}}$ (xmatch distance: 1 arcsec, odds > {photoz_odds})')
  ax.invert_xaxis()
  ax.legend(loc='upper left')
  ax.set_aspect('equal')
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('RA')
  ax.set_ylabel('DEC')




def make_overview_plots(
  info: ClusterInfo,
  df_photoz_radial: pd.DataFrame,
  df_specz_radial: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  df_members: pd.DataFrame,
  df_interlopers: pd.DataFrame,
  df_legacy_radial: pd.DataFrame,
  version: int,
  photoz_odds: float = 0.9,
  separated: bool = False,
  overwrite: bool = False,
  splus_only: bool = False,
  fmt: str = 'png',
  **kwargs
):
  wcs_spec =  {
    # 'CDELT1': -1.0,
    # 'CDELT2': 1.0,
    # 'CRPIX1': 8.5,
    # 'CRPIX2': 8.5,
    'CRVAL1': info.ra,
    'CRVAL2': info.dec,
    'CTYPE1': 'RA---AIT',
    'CTYPE2': 'DEC--AIT',
    'CUNIT1': 'deg',
    'CUNIT2': 'deg'
  }
  wcs = WCS(wcs_spec)
  
  title = _get_plot_title(info)
  
  if separated:
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'specz.{fmt}'
    with cond_overwrite(out, overwrite):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot(projection=wcs)
      _plot_specz(
        info=info,
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_specz_radial=df_specz_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'photoz.{fmt}'
    with cond_overwrite(out, overwrite):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot(projection=wcs)
      _plot_photoz(
        info=info,
        df_photoz_radial=df_photoz_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'legacy_coverage.{fmt}'
    with cond_overwrite(out, overwrite):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot(projection=wcs)
      plot_legacy_coverage(
        info=info,
        df_legacy_radial=df_legacy_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'photoz_specz.{fmt}'
    with cond_overwrite(out, overwrite):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot(projection=wcs)
      _plot_photoz_specz(
        info=info,
        df_specz_radial=df_specz_radial,
        df_photoz_radial=df_photoz_radial,
        df_all_radial=df_all_radial,
        photoz_odds=photoz_odds,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
  else:
    out_path = configs.PLOTS_FOLDER / f'cls_{info.name}.{fmt}'
    with cond_overwrite(out_path, overwrite):
      if splus_only and len(df_photoz_radial) == 0:
        return
      
      fig, axs = plt.subplots(
        nrows=3, 
        ncols=1, 
        figsize=(12, 27), 
        subplot_kw={'projection': wcs}, 
        dpi=300
      )
      
      _plot_specz(
        info=info,
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_specz_radial=df_specz_radial,
        ax=axs[0],
      )
      
      _plot_photoz(
        info=info,
        df_photoz_radial=df_photoz_radial,
        ax=axs[1],
      )
      
      _plot_photoz_specz(
        info=info,
        df_specz_radial=df_specz_radial,
        df_photoz_radial=df_photoz_radial,
        df_all_radial=df_all_radial,
        photoz_odds=photoz_odds,
        ax=axs[2],
      )
      
      fig.suptitle(title, size=18)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      










def _contour_plot(
  info: ClusterInfo,
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame,
  df_specz_radial: pd.DataFrame,
  ax: plt.Axes,
  use_photoz: bool = False
):
  dfm = radial_search(info.coord, df_members, 5 * info.r200_deg * u.deg)
  dfi = radial_search(info.coord, df_interlopers, 5 * info.r200_deg * u.deg)
  dfs = df_specz_radial
  z = dfm.z.values if not use_photoz else dfm.zml.values
  
  circle = Circle(
    (0, 0), 
    5,
    fc='none', 
    lw=2, 
    linestyle='-',
    ec='tab:green',
    label='5$\\times$R200',
  )
  ax.add_patch(circle)
  circle = Circle(
    (0, 0), 
    5*(info.r500_deg/info.r200_deg),
    fc='none', 
    lw=2, 
    linestyle='--',
    ec='tab:green',
    label='5$\\times$R500',
  )
  ax.add_patch(circle)
  xm = (dfm.ra - info.ra) / info.r200_deg
  ym = (dfm.dec - info.dec) / info.r200_deg
  mask = xm**2 + ym**2 < 5**2
  xm = xm[mask]
  ym = ym[mask]
  zm = z[mask]
  ax.scatter(
    xm, 
    ym, 
    c=zm,
    cmap='Blues', 
    s=6,
    label=f'Members ({len(dfm)})',
    rasterized=True,
    zorder=99,
  )
  xi = (dfi.ra - info.ra) / info.r200_deg
  yi = (dfi.dec - info.dec) / info.r200_deg
  mask = xi**2 + yi**2 < 5**2
  xi = xi[mask]
  yi = yi[mask]
  ax.scatter(
    xi, 
    yi, 
    marker='v',
    c='tab:gray', 
    s=6,
    label=f'Interlopers ({len(dfi)})',
    rasterized=True,
  )
  ax.scatter(
    0, 0,
    marker='+', 
    linewidths=1.5, 
    s=80, 
    c='k', 
    rasterized=True,
  )
  
  sns.kdeplot(x=xm, y=ym, levels=6, ax=ax)
  # triang = tri.Triangulation((dfm[ra_col] - info.ra) / info.r200_deg, (dfm[dec_col] - info.dec) / info.r200_deg)
  # interpolator = tri.LinearTriInterpolator(triang, dfm.z.values)
  # xi = np.linspace(-5, 5, 1000)
  # yi = np.linspace(-5, 5, 1000)
  # Xi, Yi = np.meshgrid(xi, yi)
  # zi = gaussian_filter(interpolator(Xi, Yi), 2.5)
  # ax.contour(
  #   xi, yi, zi, 
  #   levels=3, 
  #   linewidths=0.5, 
  #   colors='k',
  #   alpha=0.5,
  #   nchunk=0,
  #   corner_mask=False,
  # )
  # cmap = plt.cm.get_cmap("Blues")
  # cmap.set_under('white')
  # cmap.set_over('blue', alpha=0.6)
  # cntr1 = ax.contourf(
  #   xi, yi, zi, 
  #   levels=3, 
  #   cmap=cmap, 
  #   alpha=0.3, 
  #   nchunk=0, 
  #   corner_mask=False,
  #   vmin=dfm.z.min(),
  #   vmax=dfm.z.max()
  # )
  # ax.figure.colorbar(cntr1, ax=ax)
  
  ax.invert_xaxis()
  ax.set_aspect('equal', adjustable='datalim', anchor='C')
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.legend(loc='lower left')
  ax.set_xlabel('$\\Delta$RA/R200')
  ax.set_ylabel('$\\Delta$DEC/R200')
  ax.set_title('KDE Plot (spectroscopic members)')





def make_contour_plots(
  info: ClusterInfo,
  df_members: pd.DataFrame,
  df_interlopers: pd.DataFrame, 
  df_specz_radial: pd.DataFrame,
  version: int,
  fmt: str = 'png',
  overwrite: bool = False,
  **kwargs
):
  out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'specz_contours.{fmt}'
  if overwrite or not out.exists():
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
    ax = fig.add_subplot()
    _contour_plot(
      info=info,
      df_members=df_members,
      df_interlopers=df_interlopers,
      df_specz_radial=df_specz_radial,
      ax=ax,
      use_photoz=False,
    )
    plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
      







def _diagonal_plot(
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  df_photoz_radial: pd.DataFrame | None,
  ax: plt.Axes,
):
  if df_members is not None and df_interlopers is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    members_match = fast_crossmatch(df_members, df_all_radial)
    members_match = members_match[~members_match.z.isna() & ~members_match.zml.isna()]
    if len(members_match) == 0: return
    ax.scatter(members_match.z, members_match.zml - members_match.z, c='tab:red', s=5, alpha=0.85, label='Members', rasterized=True)
    
    interlopers_match = fast_crossmatch(df_interlopers, df_all_radial)
    interlopers_match = interlopers_match[~interlopers_match.z.isna() & ~interlopers_match.zml.isna()]
    if len(interlopers_match) == 0: return
    ax.scatter(interlopers_match.z, interlopers_match.zml - interlopers_match.z, c='tab:blue', s=5, alpha=0.85, label='Interlopers', rasterized=True)
  elif df_all_radial is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    df = df_all_radial[~df_all_radial.z.isna() & ~df_all_radial.zml.isna()]
    if len(df) == 0: return
    ax.scatter(df.z, df.zml - df.z, c='tab:blue', s=5, alpha=0.85, label='Objects', rasterized=True)
  ax.legend()
  ax.tick_params(direction='in')
  ax.set_xlabel('$z_{{spec}}$')
  ax.set_ylabel('$z_{{photo}} - z_{{spec}}$ ')
  ax.set_title('$z_{{spec}}$ x $z_{{photo}}$')
  ax.grid('on', color='k', linestyle='--', alpha=.25)



def _spec_diff_mag_plot(
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  df_photoz_radial: pd.DataFrame | None,
  ax: plt.Axes,
):
  if df_members is not None and df_interlopers is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    members_match = fast_crossmatch(df_members, df_all_radial)
    members_match = members_match[~members_match.z.isna() & ~members_match.zml.isna()]
    if len(members_match) == 0: return
    ax.scatter(members_match.r_auto, members_match.zml - members_match.z, c='tab:red', s=5, alpha=0.85, label='Members', rasterized=True)
    
    interlopers_match = fast_crossmatch(df_interlopers, df_all_radial)
    interlopers_match = interlopers_match[~interlopers_match.z.isna() & ~interlopers_match.zml.isna()]
    if len(interlopers_match) == 0: return
    ax.scatter(interlopers_match.r_auto, interlopers_match.zml - interlopers_match.z, c='tab:blue', s=5, alpha=0.85, label='Interlopers', rasterized=True)
  elif df_all_radial is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    df = df_all_radial[~df_all_radial.z.isna() & ~df_all_radial.zml.isna()]
    if len(df) == 0: return
    ax.scatter(df.r_auto, df.zml - df.z, c='tab:blue', s=5, alpha=0.85, label='Objects', rasterized=True)
  ax.legend()
  ax.tick_params(direction='in')
  ax.set_xlabel('$r_{{auto}}$')
  ax.set_ylabel('$z_{{photo}} - z_{{spec}}$ ')
  ax.set_title('$z_{{spec}}$ x $z_{{photo}}$')
  ax.grid('on', color='k', linestyle='--', alpha=.25)




def _spec_diff_odds_plot(
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  df_photoz_radial: pd.DataFrame | None,
  ax: plt.Axes,
):
  if df_members is not None and df_interlopers is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    members_match = fast_crossmatch(df_members, df_all_radial)
    members_match = members_match[~members_match.z.isna() & ~members_match.zml.isna()]
    if len(members_match) == 0: return
    ax.scatter(members_match.odds, members_match.zml - members_match.z, c='tab:red', s=5, alpha=0.85, label='Members', rasterized=True)
    
    interlopers_match = fast_crossmatch(df_interlopers, df_all_radial)
    interlopers_match = interlopers_match[~interlopers_match.z.isna() & ~interlopers_match.zml.isna()]
    if len(interlopers_match) == 0: return
    ax.scatter(interlopers_match.odds, interlopers_match.zml - interlopers_match.z, c='tab:blue', s=5, alpha=0.85, label='Interlopers', rasterized=True)
  elif df_all_radial is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    df = df_all_radial[~df_all_radial.z.isna() & ~df_all_radial.zml.isna()]
    if len(df) == 0: return
    ax.scatter(df.odds, df.zml - df.z, c='tab:blue', s=5, alpha=0.85, label='Objects', rasterized=True)
  ax.legend()
  ax.tick_params(direction='in')
  ax.set_xlabel('odds')
  ax.set_ylabel('$z_{{photo}} - z_{{spec}}$ ')
  ax.set_title('$z_{{photo}} - z_{{spec}}$ x odds')
  ax.grid('on', color='k', linestyle='--', alpha=.25)




def _spec_diff_distance_plot(
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  df_photoz_radial: pd.DataFrame | None,
  ax: plt.Axes,
):
  if df_members is not None and df_interlopers is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    members_match = fast_crossmatch(df_members, df_all_radial)
    members_match = members_match[~members_match.z.isna() & ~members_match.zml.isna()]
    if len(members_match) == 0: return
    ax.scatter(members_match.radius_Mpc, members_match.zml - members_match.z, c='tab:red', s=5, alpha=0.85, label='Members', rasterized=True)
    
    interlopers_match = fast_crossmatch(df_interlopers, df_all_radial)
    interlopers_match = interlopers_match[~interlopers_match.z.isna() & ~interlopers_match.zml.isna()]
    if len(interlopers_match) == 0: return
    ax.scatter(interlopers_match.radius_Mpc, interlopers_match.zml - interlopers_match.z, c='tab:blue', s=5, alpha=0.85, label='Interlopers', rasterized=True)
  elif df_all_radial is not None and 'r_auto' in df_all_radial.columns and 'z' in df_all_radial.columns:
    df = df_all_radial[~df_all_radial.z.isna() & ~df_all_radial.zml.isna()]
    if len(df) == 0: return
    ax.scatter(df.radius_Mpc, df.zml - df.z, c='tab:blue', s=5, alpha=0.85, label='Objects', rasterized=True)
  ax.legend()
  ax.tick_params(direction='in')
  ax.set_xlabel('Radius [Mpc]')
  ax.set_ylabel('$z_{{photo}} - z_{{spec}}$ ')
  ax.set_title('$z_{{photo}} - z_{{spec}}$ x radius')
  ax.grid('on', color='k', linestyle='--', alpha=.25)




def _histogram_members_plot(
  df_members: pd.DataFrame, 
  df_all_radial: pd.DataFrame, 
  ax: plt.Axes
):
  members_match = fast_crossmatch(df_members, df_all_radial)
  members_match = members_match[~members_match.z.isna() & ~members_match.zml.isna()]
  if len(members_match) == 0: return
  rng = (min(members_match.z.min(), members_match.zml.min()), max(members_match.z.max(), members_match.zml.max()))
  ax.hist(members_match.z, histtype='step', bins=30, range=rng, color='tab:red', alpha=0.75, lw=2, label=f'$z_{{spec}}$ ({len(members_match.z)} objects)')
  ax.hist(members_match.zml, histtype='step', bins=30, range=rng, color='tab:blue', alpha=0.75, lw=2, label=f'$z_{{photo}}$ ({len(members_match.z)} objects)')
  ax.legend()
  ax.tick_params(direction='in')
  ax.set_xlabel('z')
  ax.set_ylabel('Count (%)')
  ax.set_title('$z_{{spec}}$ x $z_{{photo}}$ (Members)')



def _histogram_interlopers_plot(
  df_interlopers: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  ax: plt.Axes,
):
  interlopers_match = fast_crossmatch(df_interlopers, df_all_radial)
  interlopers_match = interlopers_match[~interlopers_match.z.isna() & ~interlopers_match.zml.isna()]
  if len(interlopers_match) == 0: return
  rng = (min(interlopers_match.z.min(), interlopers_match.zml.min()), max(interlopers_match.z.max(), interlopers_match.zml.max()))
  ax.hist(interlopers_match.z, histtype='step', bins=30, range=rng, color='tab:red', alpha=0.75, lw=2, label=f'$z_{{spec}}$ ({len(interlopers_match.z)} objects)')
  ax.hist(interlopers_match.zml, histtype='step', bins=30, range=rng, color='tab:blue', alpha=0.75, lw=2, label=f'$z_{{photo}}$ ({len(interlopers_match.z)} objects)')
  ax.legend()
  ax.tick_params(direction='in')
  ax.set_xlabel('z')
  ax.set_ylabel('Count (%)')
  ax.set_title('$z_{{spec}}$ x $z_{{photo}}$ (Interlopers)')



def _histogram_plot_all(
  df_all_radial: pd.DataFrame, 
  ax: plt.Axes
):
  df = df_all_radial[~df_all_radial.z.isna() & ~df_all_radial.zml.isna()]
  if len(df) == 0: return
  rng = (min(df.z.min(), df.zml.min()), max(df.z.max(), df.zml.max()))
  ax.hist(df.z, histtype='step', bins=30, range=rng, color='tab:red', alpha=0.75, lw=2, label=f'$z_{{spec}}$ ({len(df.z)} objects)')
  ax.hist(df.zml, histtype='step', bins=30, range=rng, color='tab:blue', alpha=0.75, lw=2, label=f'$z_{{photo}}$ ({len(df.zml)} objects)')
  ax.legend()
  ax.tick_params(direction='in')
  ax.set_xlabel('z')
  ax.set_ylabel('Count (%)')
  ax.set_title('$z_{{spec}}$ x $z_{{photo}}$ (All)')





def make_histogram_plots(
  info: ClusterInfo,
  df_members: pd.DataFrame,
  df_interlopers: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  df_photoz_radial: pd.DataFrame | None,
  version: int,
  fmt: str = 'png',
  overwrite: bool = False,
  **kwargs
):
  if len(df_photoz_radial) > 0:
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'redshift_diagonal.{fmt}'
    if not out.exists() or overwrite:
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      _diagonal_plot(
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_all_radial=df_all_radial,
        df_photoz_radial=df_photoz_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
    
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'redshift_diff_mag.{fmt}'
    if not out.exists() or overwrite:
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      _spec_diff_mag_plot(
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_all_radial=df_all_radial,
        df_photoz_radial=df_photoz_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'redshift_diff_distance.{fmt}'
    if not out.exists() or overwrite:
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      _spec_diff_distance_plot(
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_all_radial=df_all_radial,
        df_photoz_radial=df_photoz_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
    
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'redshift_diff_odds.{fmt}'
    if not out.exists() or overwrite:
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      _spec_diff_odds_plot(
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_all_radial=df_all_radial,
        df_photoz_radial=df_photoz_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
  
  if df_members is not None and len(df_members) > 0:
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'redshift_histogram_members.{fmt}'
    if not out.exists() or overwrite:
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      _histogram_members_plot(
        df_members=df_members,
        df_all_radial=df_all_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
  if df_interlopers is not None and len(df_interlopers) > 0:
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'redshift_histogram_interlopers.{fmt}'
    if not out.exists() or overwrite:
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      _histogram_interlopers_plot(
        df_interlopers=df_interlopers,
        df_all_radial=df_all_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
  
  if df_all_radial is not None and len(df_all_radial) > 0:
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'redshift_histogram_all.{fmt}'
    if not out.exists() or overwrite:
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      _histogram_plot_all(df_all_radial=df_all_radial, ax=ax)
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)






def plot_velocity(
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame, 
  ax: plt.Axes
):
  ax.scatter(df_members.radius_Mpc, df_members.v_offset, c='tab:red', s=5, label='Members', rasterized=True)  
  ax.scatter(df_interlopers.radius_Mpc, df_interlopers.v_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
  ax.legend()
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('R [Mpc]')
  ax.set_ylabel('$\\Delta v [km/s]$')
  ax.set_title('Spectroscoptic velocity x distance')


def plot_specz(
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame, 
  cls_z: float, 
  ax: plt.Axes
):
  df_members['z_offset'] = df_members['z'] - cls_z
  df_interlopers['z_offset'] = df_interlopers['z'] - cls_z
  ax.scatter(df_members.radius_Mpc, df_members.z_offset, c='tab:red', s=5, label='Members', rasterized=True)  
  ax.scatter(df_interlopers.radius_Mpc, df_interlopers.z_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
  ax.legend()
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('R [Mpc]')
  ax.set_ylabel('$\\Delta z_{{spec}}$')
  ax.set_title('Spectroscoptic redshift x distance')
  ax.set_ylim(-0.03, 0.03)


def plot_photoz(
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame, 
  df_photoz_radial: pd.DataFrame, 
  cls_z: float,
  ax: plt.Axes,
  photoz_odds: float = 0.9, 
):
  if len(df_photoz_radial) > 0:
    df_members_match = fast_crossmatch(df_members, df_photoz_radial)
    df_interlopers_match = fast_crossmatch(df_interlopers, df_photoz_radial)
    df_members_match['zml_offset'] = df_members_match['zml'] - cls_z
    df_interlopers_match['zml_offset'] = df_interlopers_match['zml'] - cls_z
    df_members_match2 = df_members_match[df_members_match['odds'] > photoz_odds]
    df_interlopers_match2 = df_interlopers_match[df_interlopers_match['odds'] > photoz_odds]
    ax.scatter(df_members_match2.radius_Mpc, df_members_match2.zml_offset, c='tab:red', s=5, label='Members', rasterized=True)  
    ax.scatter(df_interlopers_match2.radius_Mpc, df_interlopers_match2.zml_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
  ax.legend()
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('R [Mpc]')
  ax.set_ylabel('$\\Delta z_{{photo}}$')
  ax.set_title(f'Photometric redshift x distance (odds > {photoz_odds})')
  ax.set_ylim(-0.03, 0.03)


def plot_ra_dec(
  info: ClusterInfo,
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame, 
  ax: plt.Axes
):
  _add_all_circles(info, ax)
  ax.scatter(
    df_members.ra, 
    df_members.dec, 
    c='tab:red', 
    s=5,
    label=f'Members ({len(df_members)})', 
    transform=ax.get_transform('icrs'), 
    rasterized=True,
  )
  ax.scatter(
    df_interlopers.ra, 
    df_interlopers.dec, 
    c='tab:blue', 
    s=5,
    label=f'Interlopers ({len(df_interlopers)})', 
    transform=ax.get_transform('icrs'), 
    rasterized=True,
  )
  _add_cluster_center(info.ra, info.dec, ax)
  ax.invert_xaxis()
  ax.set_aspect('equal', adjustable='datalim', anchor='C')
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.legend()
  ax.set_xlabel('RA')
  ax.set_ylabel('DEC')
  ax.set_title('Spatial distribution of spectroscopic members')
  

def plot_ra_dec_relative(
  info: ClusterInfo,
  df_members: pd.DataFrame, 
  df_interlopers: pd.DataFrame, 
  ax: plt.Axes
):
  circle = Circle(
    (0, 0), 
    5,
    fc='none', 
    lw=2, 
    linestyle='-',
    ec='tab:green',
    label='5$\\times$R200',
  )
  ax.add_patch(circle)
  circle = Circle(
    (0, 0), 
    5*(info.r500_deg/info.r200_deg),
    fc='none', 
    lw=2, 
    linestyle='--',
    ec='tab:green',
    label='5$\\times$R500',
  )
  ax.add_patch(circle)
  circle = Circle(
    (0, 0), 
    info.search_radius_deg/info.r200_deg,
    fc='none', 
    lw=2, 
    linestyle='-',
    ec='tab:brown',
    label=f'{info.search_radius_Mpc:.2f}Mpc',
  )
  ax.add_patch(circle)
  ax.scatter(
    (df_members.ra - info.ra) / info.r200_deg, 
    (df_members.dec - info.dec) / info.r200_deg, 
    c='tab:red', 
    s=5,
    label=f'Members ({len(df_members)})', 
    # transform=ax.get_transform('icrs'), 
    rasterized=True,
  )
  ax.scatter(
    (df_interlopers.ra - info.ra) / info.r200_deg, 
    (df_interlopers.dec - info.dec) / info.r200_deg, 
    c='tab:blue', 
    s=5,
    label=f'Interlopers ({len(df_interlopers)})', 
    # transform=ax.get_transform('icrs'), 
    rasterized=True,
  )
  ax.scatter(
    0, 0,
    marker='+', 
    linewidths=1.5, 
    s=80, 
    c='k', 
    rasterized=True,
  )
  ax.invert_xaxis()
  ax.legend()
  ax.set_aspect('equal', adjustable='datalim', anchor='C')
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_xlabel('$\\Delta$RA/R200')
  ax.set_ylabel('$\\Delta$DEC/R200')
  ax.set_title('Relative spatial distribution of spectroscopic members')



def make_velocity_plots(
  info: ClusterInfo,
  df_members: pd.DataFrame,
  df_interlopers: pd.DataFrame,
  df_photoz_radial: pd.DataFrame,
  version: int,
  fmt: str = 'png',
  overwrite: bool = False,
  separated: bool = True,
  photoz_odds: float = 0.9,
  **kwargs
):
  wcs_spec =  {
    # 'CDELT1': -1.0,
    # 'CDELT2': 1.0,
    # 'CRPIX1': 8.5,
    # 'CRPIX2': 8.5,
    'CRVAL1': info.ra,
    'CRVAL2': info.dec,
    'CTYPE1': 'RA---AIT',
    'CTYPE2': 'DEC--AIT',
    'CUNIT1': 'deg',
    'CUNIT2': 'deg'
  }
  wcs = WCS(wcs_spec)
  title = _get_plot_title(info)
    
  if separated:
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'spec_velocity.{fmt}'
    with cond_overwrite(out, overwrite, mkdir=True):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      plot_velocity(df_members, df_interlopers, ax)
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'specz_distance.{fmt}'
    with cond_overwrite(out, overwrite, mkdir=True):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      plot_specz(df_members, df_interlopers, info.z, ax)
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'photoz_distance.{fmt}'
    with cond_overwrite(out, overwrite, mkdir=True):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      plot_photoz(df_members, df_interlopers, df_photoz_radial, info.z, ax, photoz_odds)
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'spec_velocity_position.{fmt}'
    with cond_overwrite(out, overwrite, mkdir=True):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot(projection=wcs)
      plot_ra_dec(
        info=info,
        df_members=df_members, 
        df_interlopers=df_interlopers, 
        ax=ax
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
      
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'spec_velocity_rel_position.{fmt}'
    with cond_overwrite(out, overwrite, mkdir=True):
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      plot_ra_dec_relative(
        info=info,
        df_members=df_members, 
        df_interlopers=df_interlopers, 
        ax=ax
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
  else:
    out_path = configs.VELOCITY_PLOTS_FOLDER / f'cls_{info.name}.{fmt}'
    with cond_overwrite(out_path, overwrite, mkdir=True):
      fig = plt.figure(figsize=(7.5, 16), dpi=300)
      ax1 = fig.add_subplot(211)
      ax2 = fig.add_subplot(212, projection=wcs)
      plot_velocity(df_members, df_interlopers, ax1)
      plot_ra_dec(
        info=info,
        df_members=df_members, 
        df_interlopers=df_interlopers, 
        ax=ax2
      )
      
      fig.suptitle(title, size=18)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)






WHITE_VIRIDIS = LinearSegmentedColormap.from_list(
  'white_viridis', 
  [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
  ], 
  N=256
)


def _plot_mag_diff(
  df: pd.DataFrame,
  mag1: str, 
  mag2: str, 
  ax: plt.Axes, 
  xlabel: str = None,
  ylabel: str = None,
  xlim: Tuple[float, float] = None,
  ylim: Tuple[float, float] = None,
):
  df_spec = df[~df.z.isna()]
  df_photo = df[df.z.isna()]
  ax.scatter(
    df_photo[mag1], 
    df_photo[mag1] - df_photo[mag2], 
    s=0.8, 
    label=f'Without Spec ({len(df_photo)})', 
    color='tab:blue', 
    alpha=0.8, 
    rasterized=True
  )
  ax.scatter(
    df_spec[mag1], 
    df_spec[mag1] - df_spec[mag2], 
    s=0.8, 
    label=f'With Spec ({len(df_spec)})', 
    color='tab:red', 
    alpha=0.8,
    rasterized=True
  )
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if ylim is not None:
    ax.set_ylim(*ylim)
  if xlim is not None:
    ax.set_xlim(*xlim)
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.legend()
  ax.set_title('SP-LS r-mag difference')



def _plot_magdiff_histogram(
  x: np.ndarray | pd.Series | pd.DataFrame,
  ax: plt.Axes,
  xlabel: str = '',
  xrange: Tuple[float, float] = None,
):
  ax.hist(x, bins=60, range=xrange)
  ax.set_xlabel(xlabel)
  ax.set_ylabel('Number of galaxies')
  if xrange is not None:
    ax.set_xlim(*xrange)
  ax.grid('on', color='k', linestyle='--', alpha=.25)
  ax.tick_params(direction='in')
  ax.set_title('SP-LS r-mag difference distribution')




def make_magdiff_plots(
  info: ClusterInfo,
  df_all_radial: pd.DataFrame, 
  version: int,
  fmt: str = 'png',
  separated: bool = False,
  overwrite: bool = False,
  **kwargs
):
  if len(set(df_all_radial.columns.tolist()) & {'r_auto', 'mag_r', 'zml'}) != 3:
    return
  
  df = df_all_radial[
    (df_all_radial.type != 'PSF') & 
    df_all_radial.r_auto.between(*configs.MAG_RANGE) & 
    df_all_radial.mag_r.between(*configs.MAG_RANGE) &
    (df_all_radial.z.between(*info.z_spec_range) | df_all_radial.z.isna()) &
    df_all_radial.zml.between(*info.z_photo_range)
  ]
  title = _get_plot_title(info)
  
  if separated:
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'mag_diff.{fmt}'
    if overwrite or not out.exists():
      out.parent.mkdir(parents=True, exist_ok=True)
      fig, axs = plt.subplots(figsize=(7.5, 7.5), dpi=150)
      _plot_mag_diff(df, 'r_auto', 'mag_r', axs, '$iDR5_r$', '$iDR5_r - LS10_r$')
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
    
    out = configs.WEBSITE_PATH / f'clusters_v{version}' / info.name / f'mag_diff_hist.{fmt}'
    if overwrite or not out.exists():
      out.parent.mkdir(parents=True, exist_ok=True)
      fig, axs = plt.subplots(figsize=(7.5, 7.5), dpi=150)
      _plot_magdiff_histogram(df['r_auto'] - df['mag_r'], axs, '$iDR5_r - LS10_r$')
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
  else:
    out_path = configs.sMAGDIFF_PLOTS_FOLDER / f'cls_{info.name}.pdf'
    if not overwrite and out_path.exists():
      return

    fig, axs = plt.subplots(
      nrows=2, 
      ncols=1, 
      figsize=(5, 9),
      dpi=300,
      # subplot_kw={'projection': 'scatter_density'}, 
    )
    _plot_mag_diff(df, 'r_auto', 'mag_r', axs[0], '$iDR5_r$', '$iDR5_r - LS10_r$')
    _plot_magdiff_histogram(df['r_auto'] - df['mag_r'], axs[1], '$iDR5_r - LS10_r$')
    plt.suptitle(title, size=10)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)



def make_plots(
  info: ClusterInfo,
  df_photoz_radial: pd.DataFrame,
  df_specz_radial: pd.DataFrame,
  df_all_radial: pd.DataFrame,
  df_members: pd.DataFrame,
  df_interlopers: pd.DataFrame,
  df_legacy_radial: pd.DataFrame,
  version: int,
  photoz_odds: float = 0.9,
  separated: bool = False,
  overwrite: bool = False,
  splus_only: bool = False,
  fmt: str = 'png',
):
  kwargs = dict(
    info=info,
    df_photoz_radial=df_photoz_radial,
    df_specz_radial=df_specz_radial,
    df_all_radial=df_all_radial,
    df_members=df_members,
    df_interlopers=df_interlopers,
    df_legacy_radial=df_legacy_radial,
    version=version,
    photoz_odds=photoz_odds,
    separated=separated,
    overwrite=overwrite,
    splus_only=splus_only,
    fmt=fmt,
  )
  
  plot_functions = [
    make_overview_plots,
    make_velocity_plots,
    make_contour_plots,
    make_histogram_plots,
    make_magdiff_plots,
  ]
  
  for f in plot_functions:
    f(**kwargs)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
from astropy import units as u
from astropy.table import Column, MaskedColumn, Table
from pylegs.io import read_table, write_table

from splusclusters.configs import configs

TABLE_DESC = {
  'cluster_name': 'Cluster name, according with the Table 1',
  'ra': 'Right ascention, epoch 2000, in degrees',
  'dec': 'Declination, epoch 2000, in degrees',
  'PROB_GAL_GAIA': 'Probability of the object of being a galaxy, according to Nakazono et al.',
  'A': 'profile RMS along the major axis, in degrees',
  'B': 'profile RMS along the minor axis, in degrees',
  'THETA': 'position angle (CCW/World-x), in degrees',
  'ELLIPTICITY': 'A_IMAGE/B_IMAGE',
  'PETRO_RADIUS': 'petrosian apertures in units of A or B',
  'FLUX_RADIUS_20': r'radius enclosing 20% of the total flux',
  'FLUX_RADIUS_50': r'radius enclosing 50% of the total flux',
  'FLUX_RADIUS_90': r'radius enclosing 90% of the total flux',
  'MU_MAX_g': 'instrumental Peak surface brightness above background in g-band',
  'MU_MAX_r': 'instrumental Peak surface brightness above background in r-band',
  'BACKGROUND_g': 'instrumental background at centroid position in g-band',
  'BACKGROUND_r': 'instrumental background at centroid position in r-band',
  's2n_g_auto': 'signal to noise ratio of g-band measurement',
  's2n_r_auto': 'signal to noise ratio of r-band measurement',
  'J0378_auto': 'S-PLUS DR5 AB-calibrated magnitude for the J0378 band with aperture auto',
  'J0395_auto': 'S-PLUS DR5 AB-calibrated magnitude for the J0395 band with aperture auto',
  'J0410_auto': 'S-PLUS DR5 AB-calibrated magnitude for the J0410 band with aperture auto',
  'J0430_auto': 'S-PLUS DR5 AB-calibrated magnitude for the J0430 band with aperture auto',
  'J0515_auto': 'S-PLUS DR5 AB-calibrated magnitude for the J0515 band with aperture auto',
  'J0660_auto': 'S-PLUS DR5 AB-calibrated magnitude for the J0660 band with aperture auto',
  'J0861_auto': 'S-PLUS DR5 AB-calibrated magnitude for the J0861 band with aperture auto',
  'g_auto': 'S-PLUS DR5 AB-calibrated magnitude for the g band with aperture auto',
  'i_auto': 'S-PLUS DR5 AB-calibrated magnitude for the i band with aperture auto',
  'r_auto': 'S-PLUS DR5 AB-calibrated magnitude for the r band with aperture auto',
  'u_auto': 'S-PLUS DR5 AB-calibrated magnitude for the u band with aperture auto',
  'z_auto': 'S-PLUS DR5 AB-calibrated magnitude for the z band with aperture auto',
  'J0378_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the J0378 band with aperture PStotal',
  'J0395_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the J0395 band with aperture PStotal',
  'J0410_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the J0410 band with aperture PStotal',
  'J0430_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the J0430 band with aperture PStotal',
  'J0515_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the J0515 band with aperture PStotal',
  'J0660_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the J0660 band with aperture PStotal',
  'J0861_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the J0861 band with aperture PStotal',
  'g_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the g band with aperture PStotal',
  'i_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the i band with aperture PStotal',
  'r_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the r band with aperture PStotal',
  'u_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the u band with aperture PStotal',
  'z_PStotal': 'S-PLUS DR5 AB-calibrated magnitude for the z band with aperture PStotal',
  'J0378_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the J0378 band with aperture of 6 arcsec',
  'J0395_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the J0395 band with aperture of 6 arcsec',
  'J0410_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the J0410 band with aperture of 6 arcsec',
  'J0430_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the J0430 band with aperture of 6 arcsec',
  'J0515_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the J0515 band with aperture of 6 arcsec',
  'J0660_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the J0660 band with aperture of 6 arcsec',
  'J0861_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the J0861 band with aperture of 6 arcsec',
  'g_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the g band with aperture of 6 arcsec',
  'i_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the i band with aperture of 6 arcsec',
  'r_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the r band with aperture of 6 arcsec',
  'u_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the u band with aperture of 6 arcsec',
  'z_aper_6': 'S-PLUS DR5 AB-calibrated magnitude for the z band with aperture of 6 arcsec',
  'e_J0378_auto': 'S-PLUS DR5 magnitude error for the J0378 band with aperture auto',
  'e_J0395_auto': 'S-PLUS DR5 magnitude error for the J0395 band with aperture auto',
  'e_J0410_auto': 'S-PLUS DR5 magnitude error for the J0410 band with aperture auto',
  'e_J0430_auto': 'S-PLUS DR5 magnitude error for the J0430 band with aperture auto',
  'e_J0515_auto': 'S-PLUS DR5 magnitude error for the J0515 band with aperture auto',
  'e_J0660_auto': 'S-PLUS DR5 magnitude error for the J0660 band with aperture auto',
  'e_J0861_auto': 'S-PLUS DR5 magnitude error for the J0861 band with aperture auto',
  'e_g_auto': 'S-PLUS DR5 magnitude error for the g band with aperture auto',
  'e_i_auto': 'S-PLUS DR5 magnitude error for the i band with aperture auto',
  'e_r_auto': 'S-PLUS DR5 magnitude error for the r band with aperture auto',
  'e_u_auto': 'S-PLUS DR5 magnitude error for the u band with aperture auto',
  'e_z_auto': 'S-PLUS DR5 magnitude error for the z band with aperture auto',
  'e_J0378_PStotal': 'S-PLUS DR5 magnitude error for the J0378 band with aperture PStotal',
  'e_J0395_PStotal': 'S-PLUS DR5 magnitude error for the J0395 band with aperture PStotal',
  'e_J0410_PStotal': 'S-PLUS DR5 magnitude error for the J0410 band with aperture PStotal',
  'e_J0430_PStotal': 'S-PLUS DR5 magnitude error for the J0430 band with aperture PStotal',
  'e_J0515_PStotal': 'S-PLUS DR5 magnitude error for the J0515 band with aperture PStotal',
  'e_J0660_PStotal': 'S-PLUS DR5 magnitude error for the J0660 band with aperture PStotal',
  'e_J0861_PStotal': 'S-PLUS DR5 magnitude error for the J0861 band with aperture PStotal',
  'e_g_PStotal': 'S-PLUS DR5 magnitude error for the g band with aperture PStotal',
  'e_i_PStotal': 'S-PLUS DR5 magnitude error for the i band with aperture PStotal',
  'e_r_PStotal': 'S-PLUS DR5 magnitude error for the r band with aperture PStotal',
  'e_u_PStotal': 'S-PLUS DR5 magnitude error for the u band with aperture PStotal',
  'e_z_PStotal': 'S-PLUS DR5 magnitude error for the z band with aperture PStotal',
  'e_J0378_aper_6': 'S-PLUS DR5 magnitude error for the J0378 band with aperture of 6 arcsec',
  'e_J0395_aper_6': 'S-PLUS DR5 magnitude error for the J0395 band with aperture of 6 arcsec',
  'e_J0410_aper_6': 'S-PLUS DR5 magnitude error for the J0410 band with aperture of 6 arcsec',
  'e_J0430_aper_6': 'S-PLUS DR5 magnitude error for the J0430 band with aperture of 6 arcsec',
  'e_J0515_aper_6': 'S-PLUS DR5 magnitude error for the J0515 band with aperture of 6 arcsec',
  'e_J0660_aper_6': 'S-PLUS DR5 magnitude error for the J0660 band with aperture of 6 arcsec',
  'e_J0861_aper_6': 'S-PLUS DR5 magnitude error for the J0861 band with aperture of 6 arcsec',
  'e_g_aper_6': 'S-PLUS DR5 magnitude error for the g band with aperture of 6 arcsec',
  'e_i_aper_6': 'S-PLUS DR5 magnitude error for the i band with aperture of 6 arcsec',
  'e_r_aper_6': 'S-PLUS DR5 magnitude error for the r band with aperture of 6 arcsec',
  'e_u_aper_6': 'S-PLUS DR5 magnitude error for the u band with aperture of 6 arcsec',
  'e_z_aper_6': 'S-PLUS DR5 magnitude error for the z band with aperture of 6 arcsec',
  'g_aper_3': 'S-PLUS DR5 AB-calibrated magnitude for the g band with aperture of 3 arcsec',
  'g_res': 'S-PLUS DR5 AB-calibrated magnitude for the g band with aperture restricted auto',
  'g_iso': 'S-PLUS DR5 AB-calibrated magnitude for the g band with aperture iso',
  'g_petro': 'S-PLUS DR5 AB-calibrated magnitude for the g band with aperture petro',
  'r_aper_3': 'S-PLUS DR5 AB-calibrated magnitude for the r band with aperture of 3 arcsec',
  'r_res': 'S-PLUS DR5 AB-calibrated magnitude for the r band with aperture restricted auto',
  'r_iso': 'S-PLUS DR5 AB-calibrated magnitude for the r band with aperture iso',
  'r_petro': 'S-PLUS DR5 AB-calibrated magnitude for the r band with aperture petro',
  'Field': 'S-PLUS DR5 field name',
  'zml': 'the single-point estimate of the photometric redshift',
  'odds': 'area of the PDF contained within an interval of 0.02 of the PDF peak',
  'pdf_weights_0': 'weights of the first gaussian component of the PDF mixture of the photometric redshift',
  'pdf_weights_1': 'weights of the second gaussian component of the PDF mixture of the photometric redshift',
  'pdf_weights_2': 'weights of the third gaussian component of the PDF mixture of the photometric redshift',
  'pdf_means_0': 'means of the first gaussian component of the PDF mixture of the photometric redshift',
  'pdf_means_1': 'means of the second gaussian component of the PDF mixture of the photometric redshift',
  'pdf_means_2': 'means of the third gaussian component of the PDF mixture of the photometric redshift',
  'pdf_stds_0': 'standard deviations of the first gaussian component of the PDF mixture of the photometric redshift',
  'pdf_stds_1': 'standard deviations of the second gaussian component of the PDF mixture of the photometric redshift',
  'pdf_stds_2': 'standard deviations of the third gaussian component of the PDF mixture of the photometric redshift',
  'z': 'spectroscopic redshift',
  'v': 'velocity',
  'v_err': 'velocity error',
  'radius_deg': 'distance to the cluster center',
  'radius_Mpc': 'distance to the cluster center',
  'v_offset': 'relative to the cluster central velocity',
  'flag_member': 'indication if the object is an spectroscopic member or not of the cluster/group. 0 = member; 1 = interloper; -1 = not considered, following Section 2',
  'type': 'Legacy Survey DR10 morphological classification',
  'shape_r': r'Legacy Survey DR10 effective radius - radius that contains 50% of the light',
  'mag_g': 'Legacy Survey DR10 g-band magnitude',
  'mag_r': 'Legacy Survey DR10 r-band magnitude',
  'mag_i': 'Legacy Survey DR10 i-band magnitude',
  'mag_z': 'Legacy Survey DR10 z-band magnitude',
  'mag_w1': 'Legacy Survey DR10 w1-band magnitude',
  'mag_w2': 'Legacy Survey DR10 w2-band magnitude',
  'mag_w3': 'Legacy Survey DR10 w3-band magnitude',
  'mag_w4': 'Legacy Survey DR10 w4-band magnitude',
  'e_z': 'error in the spectroscopic redshift',
  'class_spec': 'spectroscopic classification of the object',
  # 'subclass': None,
  # 'subsubclass': None,
  # 'original_class_spec': None,
  'source': 'catalogue from which the the spectroscopic redshift was obtained',
  'f_z': 'flag for the spectroscopic redshift quality',
}

TABLE_UNIT = {
  'ra': u.deg,
  'dec': u.deg,
  'A': u.deg,
  'B': u.deg,
  'THETA': u.deg,
  'J0378_auto': u.mag,
  'J0395_auto': u.mag,
  'J0410_auto': u.mag,
  'J0430_auto': u.mag,
  'J0515_auto': u.mag,
  'J0660_auto': u.mag,
  'J0861_auto': u.mag,
  'g_auto': u.mag,
  'i_auto': u.mag,
  'r_auto': u.mag,
  'u_auto': u.mag,
  'z_auto': u.mag,
  'J0378_PStotal': u.mag,
  'J0395_PStotal': u.mag,
  'J0410_PStotal': u.mag,
  'J0430_PStotal': u.mag,
  'J0515_PStotal': u.mag,
  'J0660_PStotal': u.mag,
  'J0861_PStotal': u.mag,
  'g_PStotal': u.mag,
  'i_PStotal': u.mag,
  'r_PStotal': u.mag,
  'u_PStotal': u.mag,
  'z_PStotal': u.mag,
  'J0378_aper_6': u.mag,
  'J0395_aper_6': u.mag,
  'J0410_aper_6': u.mag,
  'J0430_aper_6': u.mag,
  'J0515_aper_6': u.mag,
  'J0660_aper_6': u.mag,
  'J0861_aper_6': u.mag,
  'g_aper_6': u.mag,
  'i_aper_6': u.mag,
  'r_aper_6': u.mag,
  'u_aper_6': u.mag,
  'z_aper_6': u.mag,
  'e_J0378_auto': u.mag,
  'e_J0395_auto': u.mag,
  'e_J0410_auto': u.mag,
  'e_J0430_auto': u.mag,
  'e_J0515_auto': u.mag,
  'e_J0660_auto': u.mag,
  'e_J0861_auto': u.mag,
  'e_g_auto': u.mag,
  'e_i_auto': u.mag,
  'e_r_auto': u.mag,
  'e_u_auto': u.mag,
  'e_z_auto': u.mag,
  'e_J0378_PStotal': u.mag,
  'e_J0395_PStotal': u.mag,
  'e_J0410_PStotal': u.mag,
  'e_J0430_PStotal': u.mag,
  'e_J0515_PStotal': u.mag,
  'e_J0660_PStotal': u.mag,
  'e_J0861_PStotal': u.mag,
  'e_g_PStotal': u.mag,
  'e_i_PStotal': u.mag,
  'e_r_PStotal': u.mag,
  'e_u_PStotal': u.mag,
  'e_z_PStotal': u.mag,
  'e_J0378_aper_6': u.mag,
  'e_J0395_aper_6': u.mag,
  'e_J0410_aper_6': u.mag,
  'e_J0430_aper_6': u.mag,
  'e_J0515_aper_6': u.mag,
  'e_J0660_aper_6': u.mag,
  'e_J0861_aper_6': u.mag,
  'e_g_aper_6': u.mag,
  'e_i_aper_6': u.mag,
  'e_r_aper_6': u.mag,
  'e_u_aper_6': u.mag,
  'e_z_aper_6': u.mag,
  'g_aper_3': u.mag,
  'g_res': u.mag,
  'g_iso': u.mag,
  'g_petro': u.mag,
  'r_aper_3': u.mag,
  'r_res': u.mag,
  'r_iso': u.mag,
  'r_petro': u.mag,
  'v': u.km / u.second,
  'v_err': u.km / u.second,
  'radius_deg': u.deg,
  'radius_Mpc': u.Mpc,
  'v_offset': u.km / u.second,
  'shape_r': u.arcsec,
  'mag_g': u.mag,
  'mag_r': u.mag,
  'mag_i': u.mag,
  'mag_z': u.mag,
  'mag_w1': u.mag,
  'mag_w2': u.mag,
  'mag_w3': u.mag,
  'mag_w4': u.mag,
}


def make_mrt(table: str):
  df = read_table(configs.OUT_PATH / f'{table}.parquet')
  t = Table()
  for c in df.columns:
    desc = TABLE_DESC.get(c)
    if desc is not None:
      mask = df[c].isna() | (df[c] == np.inf) | (df[c] == -np.inf)
      if np.all(mask):
        continue
      elif np.any(mask):
        col = MaskedColumn(df[c].values, name=c, description=desc, unit=TABLE_UNIT.get(c), mask=mask)
      else:
        col = Column(df[c].values, name=c, description=desc, unit=TABLE_UNIT.get(c))
      t[c] = col
  t.write(configs.OUT_PATH / f'{table}.dat', format='ascii.mrt', overwrite=True)
  
if __name__ == '__main__':
  print('>> Table 3')
  make_mrt('table_3')
  print('>> Table 4')
  make_mrt('table_4')
  print('>> Table 5')
  make_mrt('table_5')
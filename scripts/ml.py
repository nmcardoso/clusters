import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import warnings
from functools import partial
from typing import Callable, Dict, List, Literal, Mapping, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from optuna.distributions import *
from pylegs.io import read_table, write_table
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBClassifier

from splusclusters.configs import configs

warnings.filterwarnings("ignore", module="lightgbm")

# TABLE_PATH = '~/Downloads/table_3.parquet'
TABLE_PATH = '~/Downloads/samples_ForNatanael.csv'

geom = [
  'A', 'B', 'THETA', 'ELLIPTICITY', 'PETRO_RADIUS', 'FLUX_RADIUS_50', 
  'FLUX_RADIUS_90', 'MU_MAX_g', 'MU_MAX_r', 'BACKGROUND_g', 'BACKGROUND_r', 
  's2n_g_auto', 's2n_r_auto', 'zml' 
]
ngeom = [ 
  'D_CENTER/R200_deg', '(A/B)','(FLUX_RADIUS_50/PETRO_RADIUS)', '(FLUX_RADIUS_90/PETRO_RADIUS)', 
  '(FLUX_RADIUS_50/PETRO_RADIUS)*(A/B)', 'Densidad_vecinos', 'r_auto/area', 'Area_Voronoi', 
  'Area_Voronoi_norm', 'Diferencia_angular'
]
bands = [
  'J0378_auto', 'J0395_auto', 'J0410_auto', 'J0430_auto', 'J0515_auto', 'J0660_auto', 
  'J0861_auto', 'g_auto', 'i_auto', 'r_auto', 'u_auto', 'z_auto'
]
bands_e = [ 
  'e_J0378_auto', 'e_J0395_auto', 'e_J0410_auto', 'e_J0430_auto', 'e_J0515_auto', 
  'e_J0660_auto', 'e_J0861_auto', 'e_g_auto', 'e_i_auto', 'e_r_auto', 'e_u_auto', 'e_z_auto' 
]
bands_PS = [ 
  'J0378_PStotal', 'J0395_PStotal', 'J0410_PStotal', 'J0430_PStotal', 'J0515_PStotal', 
  'J0660_PStotal', 'J0861_PStotal', 'g_PStotal', 'i_PStotal', 'r_PStotal', 'u_PStotal', 'z_PStotal'
] 
bands_PS_e = [
  'e_J0378_PStotal', 'e_J0395_PStotal', 'e_J0410_PStotal', 'e_J0430_PStotal', 'e_J0515_PStotal', 
  'e_J0660_PStotal', 'e_J0861_PStotal', 'e_g_PStotal', 'e_i_PStotal', 'e_r_PStotal', 'e_u_PStotal', 'e_z_PStotal'
]
band_iso = ['r_iso', 'r_petro', 'r_aper_3', 'r_aper_6']

features = bands + geom + ngeom

R200 = {
  'A168': 4.701 / 5,
  'A639': 4.927 / 5,
  'MKW1': 1.256 / 5,
  '[YMV2007]7604': 1.441 / 5,
  'HCG97': 4.952 / 5,
  'IC1860': 4.542 / 5,
  'WBL074': 4.180 / 5,
  'NGC1132': 2.065 / 5,
  'A2870': 1.989 / 5,
  'A2877': 3.394 / 5,
}

RF_HPS = {
  'n_estimators': IntDistribution(20, 400, step=10),
  'criterion': CategoricalDistribution(['gini', 'entropy', 'log_loss']),
  'max_depth': IntDistribution(0, 200),
  'min_samples_split': FloatDistribution(0, 0.5),
  'min_samples_leaf': FloatDistribution(0, 0.5),
  'min_weight_fraction_leaf': FloatDistribution(0, 0.5),
  'max_features': FloatDistribution(0, 1),
  'max_leaf_nodes': IntDistribution(2, 100),
  'min_impurity_decrease': FloatDistribution(0, 0.3),
  # 'bootstrap': CategoricalDistribution([True, False]),
  'oob_score': CategoricalDistribution([True, False]),
  'class_weight': CategoricalDistribution(['balanced', 'balanced_subsample', None]),
  'ccp_alpha': FloatDistribution(0.0, 0.4),
  'max_samples': FloatDistribution(0.1, 1.0),
  'random_state': 42,
}

LGBM_HPS = {
  'boosting_type': CategoricalDistribution(['gbdt', 'dart', 'rf']),
  'num_leaves': IntDistribution(4, 70),
  'max_depth': IntDistribution(2, 10),
  # 'min_data_in_leaf': IntDistribution(10, 40),
  'learning_rate': FloatDistribution(0.0001, 0.2, log=True),
  'n_estimators': IntDistribution(20, 400, step=10),
  # 'min_split_gain': FloatDistribution(0, 0.4),
  # 'min_child_weight': FloatDistribution(0.0001, 0.1, log=True),
  # 'min_child_samples': IntDistribution(10, 40),
  'subsample': FloatDistribution(0.7, 1.0),
  # 'subsample_freq': IntDistribution(0, 3),
  'colsample_bytree': FloatDistribution(0.8, 1.0),
  'reg_alpha': FloatDistribution(0, 0.2),
  'reg_lambda': FloatDistribution(0, 0.2),
  'importance_type': CategoricalDistribution(['split', 'gain']),
  'verbose': -1,
}

XGB_HPS = {
  'n_estimators': IntDistribution(20, 400, step=10),
  'num_leaves': IntDistribution(4, 70),
  'max_depth': IntDistribution(2, 10),
  'grow_policy': CategoricalDistribution(['depthwise', 'lossguide']),
  'booster': CategoricalDistribution(['gbtree', 'gblinear', 'dart']),
  'tree_method': CategoricalDistribution(['approx', 'hist']),
  'gamma': FloatDistribution(0, 0.2),
  'min_child_weight': FloatDistribution(0.0001, 0.1, log=True),
  'max_delta_step': FloatDistribution(0.3, 1.0),
  'subsample': FloatDistribution(0.7, 1.0),
  'subsampling_method': CategoricalDistribution(['uniform', 'gradient_based']),
  'colsample_bytree': FloatDistribution(0.6, 1.0),
  'colsample_bylevel': FloatDistribution(0.6, 1.0),
  'colsample_bynode': FloatDistribution(0.6, 1.0),
  'reg_alpha': FloatDistribution(0, 0.2),
  'reg_lambda': FloatDistribution(0, 0.2),
  'importance_type': CategoricalDistribution(['gain', 'weight', 'cover', 'total_gain', 'total_cover']),
  'verbosity': 0,
}



def make_reference_sample():
  df = read_table(TABLE_PATH)
  print(*df.columns, sep=', ')
  clusters = ['A168', 'A639', 'MKW1', '[YMV2007]7604', 'HCG97', 'IC1860', 'WBL074', 'NGC1132', 'A2870', 'A2877']
  for k, v in R200.items():
    df.loc[df.name == k, 'r200'] = v
  mask = df.name.isin(clusters) & (df.radius_Mpc < 5 * df.r200) 
  df = df[mask].reset_index(drop=True).copy()
  print(f'reference set before nan drop: {len(df)}')
  df = df.dropna(axis='index', subset=features).reset_index(drop=True).copy()
  print(f'reference set after nan drop: {len(df)}')
  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  for i, (train_idx, test_idx) in enumerate(kfold.split(df.flag_member.values, df.flag_member.values)):
    df.loc[test_idx, 'fold'] = i
    print(f'Fold {i}: members: {len(df[(df.fold == i) & (df.flag_member == 0)])}; interlopers: {len(df[(df.fold == i) & (df.flag_member == 1)])}')
  return df




def split_train_test(df: pd.DataFrame, test_fold: int = 0):
  X = df.loc[df.fold != test_fold, features].values
  y = df.loc[df.fold != test_fold, 'flag_member'].values
  X_test = df.loc[df.fold == test_fold, features].values
  y_test = df.loc[df.fold == test_fold, 'flag_member'].values
  return X, y, X_test, y_test




def train_model(
  model: Literal['rf', 'lgbm', 'xgb'],
  X: np.ndarray,
  y: np.ndarray,
  params: Dict[str, Any]
):
  model_factory = {
    'rf': RandomForestClassifier,
    'lgbm': LGBMClassifier,
    'xgb': XGBClassifier,
  }
  
  clf = model_factory[model](**params)
  clf.fit(X, y)
  
  return clf



  
def kfold(model: Literal['rf', 'lgbm', 'xgb'], df: pd.DataFrame, **kwargs):
  y_test_list, y_pred_list = [], []
  for i in range(int(df.fold.max()) + 1):
    X, y, X_test, y_test = split_train_test(df, i)
    
    clf = train_model(model, X, y, kwargs)
    y_pred = clf.predict(X_test)
    
    y_test_list.append(y_test)
    y_pred_list.append(y_pred)
  return X_test, y_test_list, y_pred_list



def compute_kfold_metrics(
  y_test: List[np.ndarray], 
  y_pred: List[np.ndarray], 
  metrics: Sequence[Literal['f1', 'precision', 'recall', 'roc', 'pr', 'accuracy']],
) -> Tuple[Dict[str, float], Dict[str, float]]:
  func_mapping = {
    'f1': f1_score,
    'precision': precision_score,
    'recall': recall_score,
    'roc': roc_auc_score,
    'pr': precision_recall_curve,
    'accuracy': accuracy_score,
  }
  metric_values = {k : [] for k in metrics}
  for y_test_fold, y_pred_fold in zip(y_test, y_pred):
    for metric in metrics:
      metric_values[metric].append(func_mapping[metric](y_test_fold, y_pred_fold))

  mean = {k: np.mean(metric_values[k]) for k in metric_values.keys()}
  std = {k: np.std(metric_values[k]) for k in metric_values.keys()}
  return mean, std



def hp_tune(
  model: Literal['rf', 'lgbm', 'xgb'], 
  df: pd.DataFrame, 
  hps: Dict[str, Any], 
  study_name: str, 
  n_trials: int = 10, 
  n_budget: int = 100
):
  sampler = optuna.samplers.TPESampler(n_startup_trials=n_budget) # recommended budget for TPE sampler: 100-1000
  storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(str((configs.STUDY_FOLDER / (study_name + '.log')).absolute()))
  )
  directions = ['maximize', 'maximize']
  
  study = optuna.create_study(
    storage=storage, 
    sampler=sampler, 
    study_name=study_name, 
    directions=directions, 
    load_if_exists=True
  )
  
  for i in range(n_trials):
    print(f'>> Trial #{i+1} of {n_trials}')
    dist_hps = {k: v for k, v in hps.items() if isinstance(v, BaseDistribution)}
    trial = study.ask(dist_hps)
    for k in set(hps.keys()) - set(dist_hps.keys()):
      trial.set_user_attr(k, hps[k])
    merged = {**trial.params, **trial.user_attrs}
    kwargs = {k: merged[k] for k in hps.keys()}
    _, y_true, y_pred = kfold(model, df, **kwargs)
    mean, std = compute_kfold_metrics(y_true, y_pred, metrics=['f1', 'roc'])
    trial.set_user_attr('f1', mean['f1'])
    trial.set_user_attr('roc', mean['roc'])
    study.tell(trial, [mean['f1'], mean['roc']])
    
    print(f'  - Current trial:')
    print('    ', end='')
    for k, v in kwargs.items():
      if isinstance(v, float):
        print(f'{k}: {v:.3f}', end=', ')
      else:
        print(f'{k}: {v}', end=', ')
    print()
    print('    ', end='')
    for k in mean.keys():
      print(f'{k}: {mean[k]:.2f} ± {std[k]:.2f}', end=', ')
    print()
    
    if (i > 0) and ((i + 1) % 10 == 0):
      print()
      print(f'  - Best trial (#{study.best_trials[0]._trial_id}):')
      print('    ', end='')
      for k, v in study.best_trials[0].params.items():
        if isinstance(v, float):
          print(f'{k}: {v:.3f}', end=', ')
        else:
          print(f'{k}: {v}', end=', ')
      print()
      m = study.best_trials[0].values
      
      print('    ', end='')
      for k in mean.keys():
        v = study.best_trials[0].user_attrs.get(k)
        if v is not None:
          print(f'{k}: {v:.2f}', end=', ')
      print()
    print()
      




def main():
  df = make_reference_sample()
  hp_tune('xgb', df, XGB_HPS, 'xgb_run_01', 4000, 800)



if __name__ == "__main__":
  main()
                
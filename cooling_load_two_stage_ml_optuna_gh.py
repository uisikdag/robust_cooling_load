import os
import sys
import time
import json
import platform
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import optuna

from sklearn.model_selection import (
    KFold, RepeatedKFold, cross_val_score,
)
from scipy import stats as scipy_stats
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

try:
    import shap
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'heat_reflective_building_envelope_dataset.csv')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results_cooling_load')
SUMMARY_DIR = os.path.join(RESULTS_DIR, 'summary')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class _Tee:

    def __init__(self, *streams):
        self._streams = streams

    def write(self, msg):
        for s in self._streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def start_logging():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f'run_{stamp}.log')
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    optuna.logging.enable_propagation()
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

    print(f"[log] writing console output to {log_path}")
    return log_file


def capture_environment():
    def _version(module_name):
        try:
            mod = __import__(module_name)
            return getattr(mod, '__version__', 'unknown')
        except Exception as e:
            return f'NOT AVAILABLE ({e})'

    env = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'cpu_count_logical': os.cpu_count(),
        'hostname': platform.node(),
        'package_versions': {
            'numpy': _version('numpy'),
            'pandas': _version('pandas'),
            'scikit-learn': _version('sklearn'),
            'optuna': _version('optuna'),
            'xgboost': _version('xgboost'),
            'lightgbm': _version('lightgbm'),
            'catboost': _version('catboost'),
            'scipy': _version('scipy'),
        },
        'seeds': {
            'CV_RANDOM_STATE': CV_RANDOM_STATE,
            'OUTER_CV_FOLDS': OUTER_CV_FOLDS,
            'OUTER_CV_REPEATS': OUTER_CV_REPEATS,
            'INNER_CV_FOLDS': INNER_CV_FOLDS,
            'N_TRIALS': N_TRIALS,
            'BOOTSTRAP_SEED': BOOTSTRAP_SEED,
            'note': 'inner Optuna study seed per outer fold = CV_RANDOM_STATE + fold_idx',
        },
        'model_n_jobs': MODEL_N_JOBS,
    }

    with open(os.path.join(SUMMARY_DIR, 'environment.json'), 'w', encoding='utf-8') as f:
        json.dump(env, f, indent=2)

    print("\n>>> Environment / provenance (see summary/environment.json):")
    print(f"    UTC timestamp   : {env['timestamp_utc']}")
    print(f"    Python          : {env['python_version']}")
    print(f"    Platform        : {env['platform']}")
    print(f"    Processor       : {env['processor']}  ({env['cpu_count_logical']} logical CPUs)")
    print(f"    Hostname        : {env['hostname']}")
    for pkg, ver in env['package_versions'].items():
        print(f"    {pkg:<15} : {ver}")
    print(f"    MODEL_N_JOBS    : {MODEL_N_JOBS}")

    return env


CV_RANDOM_STATE = 0
OUTER_CV_REPEATS = 10
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 5
N_TRIALS = 50
SAVE_FOLD_FIGURES = False

MODEL_N_JOBS = 1

BOOTSTRAP_N = 10000
BOOTSTRAP_CI = 95
BOOTSTRAP_SEED = 0

PERM_IMPORTANCE_N_REPEATS_OUTER = 10

ORDINAL_COLS = ['Insulation_Level', 'Occupancy_Level']
ORDINAL_CATEGORIES = [['Low', 'Medium', 'High'], ['Low', 'Medium', 'High']]
ONEHOT_COLS = ['Building_Orientation', 'Heat_Reflective_Treatment']
NUMERIC_COLS = [
    'Ambient_Temperature_C', 'Solar_Radiation_Wm2', 'Outdoor_Humidity_%',
    'Wind_Speed_ms', 'Emissivity', 'Wall_Thickness_cm', 'HVAC_Setpoint_C',
    'Coating_Reflectivity_%',
]
PREDICTORS = NUMERIC_COLS + ORDINAL_COLS + ONEHOT_COLS
TARGET = 'Cooling_Load_kWh'

NEEDS_SCALING = {'Ridge', 'Lasso', 'ElasticNet', 'LinearRegression',
                 'HuberRegressor', 'KNN', 'SVR', 'MLP'}
TREE_LIKE = {'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'ExtraTrees',
             'GradientBoosting', 'HistGradientBoosting', 'AdaBoost',
             'Bagging', 'DecisionTree'}
LINEAR_FAMILY = {'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'HuberRegressor'}

data = pd.read_csv(DATA_PATH)
data = data[PREDICTORS + [TARGET]].copy()

X_df = data[PREDICTORS].copy()
y_all = data[TARGET].values.astype(np.float64)

ONEHOT_CATEGORIES = [sorted(X_df[c].dropna().unique().tolist()) for c in ONEHOT_COLS]

print("Target: {} (n={}, mean={:.2f}, std={:.2f})".format(
    TARGET, len(y_all), y_all.mean(), y_all.std()))
print("Predictors ({}): {}".format(len(PREDICTORS), PREDICTORS))
print("SHAP available: {}".format(_HAVE_SHAP))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_ci(values, n_boot=BOOTSTRAP_N, ci=BOOTSTRAP_CI, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, alpha))
    hi = float(np.percentile(boot_means, 100.0 - alpha))
    return lo, hi


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', 'passthrough', NUMERIC_COLS),
            ('ord', OrdinalEncoder(categories=ORDINAL_CATEGORIES,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1), ORDINAL_COLS),
            ('ohe', OneHotEncoder(categories=ONEHOT_CATEGORIES, drop='first',
                                  handle_unknown='ignore',
                                  sparse_output=False), ONEHOT_COLS),
        ],
        remainder='drop',
    )


def preprocessor_feature_names(preprocessor):
    names = list(NUMERIC_COLS) + list(ORDINAL_COLS)
    ohe = preprocessor.named_transformers_['ohe']
    names += list(ohe.get_feature_names_out(ONEHOT_COLS))
    return names


def build_pipeline(name, estimator):
    steps = [('prep', make_preprocessor())]
    if name in NEEDS_SCALING:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', estimator))
    return Pipeline(steps)


def build_model_zoo(seed):
    return {
        'XGBoost': XGBRegressor(random_state=seed, n_jobs=MODEL_N_JOBS,
                                tree_method='hist',
                                objective='reg:squarederror'),
        'LightGBM': LGBMRegressor(random_state=seed, n_jobs=MODEL_N_JOBS, verbose=-1),
        'CatBoost': CatBoostRegressor(random_state=seed, verbose=0,
                                      thread_count=MODEL_N_JOBS,
                                      allow_writing_files=False),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=seed),
        'GradientBoosting': GradientBoostingRegressor(random_state=seed),
        'AdaBoost': AdaBoostRegressor(random_state=seed),
        'RandomForest': RandomForestRegressor(random_state=seed, n_jobs=MODEL_N_JOBS),
        'ExtraTrees': ExtraTreesRegressor(random_state=seed, n_jobs=MODEL_N_JOBS),
        'Bagging': BaggingRegressor(random_state=seed, n_jobs=MODEL_N_JOBS),
        'DecisionTree': DecisionTreeRegressor(random_state=seed),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=seed),
        'Lasso': Lasso(random_state=seed),
        'ElasticNet': ElasticNet(random_state=seed),
        'HuberRegressor': HuberRegressor(max_iter=500),
        'KNN': KNeighborsRegressor(n_jobs=MODEL_N_JOBS),
        'SVR': SVR(),
        'MLP': MLPRegressor(random_state=seed, max_iter=500),
    }


def suggest_params(name, trial):
    if name == 'XGBoost':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            gamma=trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        )
    if name == 'LightGBM':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
            num_leaves=trial.suggest_int('num_leaves', 15, 255),
            max_depth=trial.suggest_int('max_depth', -1, 12),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 60),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        )
    if name == 'CatBoost':
        return dict(
            iterations=trial.suggest_int('iterations', 100, 800, step=50),
            depth=trial.suggest_int('depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
        )
    if name in ('HistGradientBoosting',):
        return dict(
            max_iter=trial.suggest_int('max_iter', 100, 600, step=50),
            max_depth=trial.suggest_int('max_depth', 2, 12),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            l2_regularization=trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
            max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 255),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 60),
        )
    if name == 'GradientBoosting':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
            max_depth=trial.suggest_int('max_depth', 2, 8),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 30),
        )
    if name in ('RandomForest', 'ExtraTrees'):
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
            max_depth=trial.suggest_int('max_depth', 3, 25),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_float('max_features', 0.3, 1.0),
        )
    if name == 'AdaBoost':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 50, 500, step=50),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 2.0, log=True),
        )
    if name == 'Bagging':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 10, 200, step=10),
            max_samples=trial.suggest_float('max_samples', 0.5, 1.0),
            max_features=trial.suggest_float('max_features', 0.5, 1.0),
        )
    if name == 'DecisionTree':
        return dict(
            max_depth=trial.suggest_int('max_depth', 2, 25),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        )
    if name == 'Ridge':
        return dict(alpha=trial.suggest_float('alpha', 1e-3, 100.0, log=True))
    if name == 'Lasso':
        return dict(alpha=trial.suggest_float('alpha', 1e-4, 10.0, log=True))
    if name == 'ElasticNet':
        return dict(
            alpha=trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            l1_ratio=trial.suggest_float('l1_ratio', 0.0, 1.0),
        )
    if name == 'HuberRegressor':
        return dict(
            alpha=trial.suggest_float('alpha', 1e-5, 1.0, log=True),
            epsilon=trial.suggest_float('epsilon', 1.1, 3.0),
        )
    if name == 'LinearRegression':
        return dict()
    if name == 'KNN':
        return dict(
            n_neighbors=trial.suggest_int('n_neighbors', 3, 40),
            weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
            p=trial.suggest_int('p', 1, 2),
        )
    if name == 'SVR':
        return dict(
            C=trial.suggest_float('C', 1e-1, 1e3, log=True),
            gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
            epsilon=trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
        )
    if name == 'MLP':
        return dict(
            hidden_layer_sizes=trial.suggest_categorical(
                'hidden_layer_sizes', [(64,), (128,), (64, 32), (128, 64)]),
            alpha=trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        )
    return dict()


def build_estimator(name, params, seed):
    if name == 'XGBoost':
        return XGBRegressor(random_state=seed, n_jobs=MODEL_N_JOBS, tree_method='hist',
                            objective='reg:squarederror', **params)
    if name == 'LightGBM':
        return LGBMRegressor(random_state=seed, n_jobs=MODEL_N_JOBS, verbose=-1, **params)
    if name == 'CatBoost':
        return CatBoostRegressor(random_state=seed, verbose=0,
                                 thread_count=MODEL_N_JOBS,
                                 allow_writing_files=False, **params)
    if name == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(random_state=seed, **params)
    if name == 'GradientBoosting':
        return GradientBoostingRegressor(random_state=seed, **params)
    if name == 'AdaBoost':
        return AdaBoostRegressor(random_state=seed, **params)
    if name == 'RandomForest':
        return RandomForestRegressor(random_state=seed, n_jobs=MODEL_N_JOBS, **params)
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(random_state=seed, n_jobs=MODEL_N_JOBS, **params)
    if name == 'Bagging':
        return BaggingRegressor(random_state=seed, n_jobs=MODEL_N_JOBS, **params)
    if name == 'DecisionTree':
        return DecisionTreeRegressor(random_state=seed, **params)
    if name == 'LinearRegression':
        return LinearRegression(**params)
    if name == 'Ridge':
        return Ridge(random_state=seed, **params)
    if name == 'Lasso':
        return Lasso(random_state=seed, **params)
    if name == 'ElasticNet':
        return ElasticNet(random_state=seed, **params)
    if name == 'HuberRegressor':
        return HuberRegressor(max_iter=1000, **params)
    if name == 'KNN':
        return KNeighborsRegressor(n_jobs=MODEL_N_JOBS, **params)
    if name == 'SVR':
        return SVR(**params)
    if name == 'MLP':
        return MLPRegressor(random_state=seed, max_iter=800, **params)
    raise ValueError(f"Unknown model {name}")


def build_tuned(name, params, seed):
    return build_pipeline(name, build_estimator(name, params, seed))


def get_feature_importance(fitted_pipe):
    model = fitted_pipe.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        return np.asarray(model.feature_importances_, dtype=float)
    return None


def optuna_convergence_stats(study, tol_pct=1.0):
    values = np.array([t.value for t in study.trials], dtype=float)
    running_best_rmse = -np.maximum.accumulate(values)
    n_trials = len(running_best_rmse)
    final_rmse = float(running_best_rmse[-1])
    first_rmse = float(running_best_rmse[0])
    threshold = final_rmse * (1.0 + tol_pct / 100.0)
    trial_within_tol = int(np.argmax(running_best_rmse <= threshold)) + 1

    q3 = max(1, int(np.floor(0.75 * n_trials)))
    total_improvement = first_rmse - final_rmse
    late_improvement = running_best_rmse[q3 - 1] - final_rmse
    pct_improvement_in_last_quarter = (
        100.0 * late_improvement / total_improvement if total_improvement > 1e-12 else 0.0
    )

    return {
        'n_trials': n_trials,
        'first_trial_rmse': first_rmse,
        'final_rmse': final_rmse,
        f'trial_reaching_within_{tol_pct:.0f}pct_of_final': trial_within_tol,
        'pct_improvement_in_last_quarter_of_trials': float(pct_improvement_in_last_quarter),
        'running_best_rmse': running_best_rmse,
    }


def tune_on_partition(name, X_part, y_part, seed):
    cv = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        params = suggest_params(name, trial)
        pipe = build_tuned(name, params, seed)
        scores = cross_val_score(pipe, X_part, y_part, cv=cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
        return scores.mean()

    if name == 'LinearRegression':
        cv_rmse = -cross_val_score(
            build_tuned(name, {}, seed), X_part, y_part, cv=cv,
            scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return {}, float(cv_rmse), None

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    conv = optuna_convergence_stats(study, tol_pct=1.0)
    return study.best_params, -study.best_value, conv


def plot_results(prefix, y_true, y_pred, r2_, rmse_, mae_, mape_,
                 fi_df, color, filename):
    resid = y_true - y_pred
    plt.figure(figsize=(18, 12))

    ax1 = plt.subplot(2, 3, 1)
    if fi_df is not None:
        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax1)
        ax1.set_title(f'{prefix} - Feature Importance')
    else:
        ax1.axis('off')
        ax1.set_title(f'{prefix} - (no native importances)')

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_true, y_pred, s=8, alpha=0.3, color=color)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([lo, hi], [lo, hi], 'k--', alpha=0.6)
    ax2.set_xlabel('Actual (kWh)'); ax2.set_ylabel('Predicted (kWh)')
    ax2.set_title(f'{prefix} - Predicted vs Actual'); ax2.grid(alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    names = ['R2', 'RMSE', 'MAE', 'MAPE']
    vals = [r2_, rmse_, mae_, mape_]
    bars = ax3.bar(names, vals, color='steelblue')
    ax3.grid(alpha=0.3, axis='y'); ax3.set_title(f'{prefix} - Metrics')
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width()/2, v, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9)

    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(y_pred, resid, s=8, alpha=0.3, color=color)
    ax4.axhline(0, color='k', ls='--', alpha=0.6)
    ax4.set_xlabel('Predicted (kWh)'); ax4.set_ylabel('Residual (kWh)')
    ax4.set_title(f'{prefix} - Residuals vs Predicted'); ax4.grid(alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(resid, bins=40, alpha=0.75, color=color)
    ax5.axvline(0, color='k', ls='--', alpha=0.6)
    ax5.set_xlabel('Residual (kWh)'); ax5.set_ylabel('Count')
    ax5.set_title(f'{prefix} - Residual Distribution'); ax5.grid(alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(y_true, np.abs(resid), s=8, alpha=0.3, color=color)
    ax6.set_xlabel('Actual (kWh)'); ax6.set_ylabel('|Residual| (kWh)')
    ax6.set_title(f'{prefix} - Absolute Error vs Actual'); ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_aggregate(prefix, metrics_dict, fi_df_, color, filename, n_folds):
    plt.figure(figsize=(18, 10))
    metric_keys = ['r2', 'rmse', 'mae', 'mape']
    nice = ['R2', 'RMSE', 'MAE', 'MAPE']
    tag = f'{n_folds} outer folds'

    ax1 = plt.subplot(2, 2, 1)
    if fi_df_ is not None:
        sns.barplot(x='Importance', y='Feature', data=fi_df_, ax=ax1)
        ax1.set_title(f'{prefix} - Mean Feature Importance ({tag})')
    else:
        ax1.axis('off'); ax1.set_title(f'{prefix} - (no native importances)')

    ax2 = plt.subplot(2, 2, 2)
    means = [np.mean(metrics_dict[k]) for k in metric_keys]
    stds = [np.std(metrics_dict[k]) for k in metric_keys]
    xpos = np.arange(len(nice))
    bars = ax2.bar(xpos, means, yerr=stds, capsize=4, color='steelblue')
    ax2.set_xticks(xpos); ax2.set_xticklabels(nice)
    ax2.grid(alpha=0.3, axis='y'); ax2.set_title(f'{prefix} - Mean +/- Std ({tag})')
    for b, m, s in zip(bars, means, stds):
        ax2.text(b.get_x() + b.get_width()/2, m + s, f'{m:.3f}\n+/-{s:.3f}',
                 ha='center', va='bottom', fontsize=8)

    ax3 = plt.subplot(2, 2, 3)
    ax3.boxplot([metrics_dict['r2']])
    ax3.set_xticklabels(['R2'])
    ax3.scatter(np.ones(len(metrics_dict['r2'])), metrics_dict['r2'],
                alpha=0.4, color=color)
    ax3.set_title(f'{prefix} - Per-Fold R2'); ax3.grid(alpha=0.3, axis='y')

    ax4 = plt.subplot(2, 2, 4)
    ax4.boxplot([metrics_dict['rmse'], metrics_dict['mae']])
    ax4.set_xticklabels(['RMSE', 'MAE'])
    ax4.set_title(f'{prefix} - Per-Fold Error (kWh)'); ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ci(prefix, metrics_dict, filename, model_name):
    metric_keys = ['r2', 'rmse', 'mae', 'mape']
    nice = ['R2', 'RMSE (kWh)', 'MAE (kWh)', 'MAPE']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, k, n in zip(axes, metric_keys, nice):
        vals = np.asarray(metrics_dict[k], dtype=float)
        mean = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals)
        ax.scatter(np.random.default_rng(0).normal(1, 0.03, vals.size), vals,
                   alpha=0.35, color='seagreen', s=18, zorder=1)
        ax.errorbar([1], [mean], yerr=[[mean - lo], [hi - mean]], fmt='o',
                    color='black', capsize=8, capthick=2, elinewidth=2,
                    markersize=8, zorder=3,
                    label=f'mean {mean:.3f}\n{BOOTSTRAP_CI}% CI [{lo:.3f}, {hi:.3f}]')
        ax.set_xticks([]); ax.set_title(n)
        ax.grid(alpha=0.3, axis='y'); ax.legend(loc='best', fontsize=8)
    fig.suptitle(f'{model_name} - {prefix}: mean +/- {BOOTSTRAP_CI}% bootstrap CI '
                 f'over outer folds', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def breusch_pagan_test(y_pred, resid):
    n = len(resid)
    sq_resid = resid ** 2
    aux = LinearRegression().fit(y_pred.reshape(-1, 1), sq_resid)
    aux_r2 = aux.score(y_pred.reshape(-1, 1), sq_resid)
    lm_stat = n * aux_r2
    p_value = float(scipy_stats.chi2.sf(lm_stat, df=1))
    return float(lm_stat), p_value, float(aux_r2)


def plot_residual_diagnostics(name, oof_df, out_dir):
    y_true = oof_df['y_true'].to_numpy()
    y_pred = oof_df['y_pred'].to_numpy()
    resid = y_true - y_pred
    std_resid = resid / (resid.std() + 1e-12)

    lm_stat, bp_p, aux_r2 = breusch_pagan_test(y_pred, resid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, s=6, alpha=0.15, color='seagreen')
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.7, label='1:1 (perfect prediction)')
    ax.set_xlabel('Actual Cooling_Load_kWh'); ax.set_ylabel('Predicted (nested-CV, out-of-fold)')
    ax.set_title(f'{name}: Predicted vs Actual (out-of-fold, {len(y_true)} points)')
    ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y_pred, resid, s=6, alpha=0.15, color='seagreen')
    ax.axhline(0, color='k', ls='--', alpha=0.7)
    ax.set_xlabel('Predicted (kWh)'); ax.set_ylabel('Residual = Actual - Predicted (kWh)')
    ax.set_title('Residuals vs Predicted (heteroscedasticity check)')
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, f'Breusch-Pagan: LM={lm_stat:.2f}, p={bp_p:.4f}'
                        f'\n(p<0.05 => evidence of heteroscedasticity)',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[1, 0]
    sqrt_abs_std_resid = np.sqrt(np.abs(std_resid))
    ax.scatter(y_pred, sqrt_abs_std_resid, s=6, alpha=0.15, color='seagreen')
    ax.set_xlabel('Predicted (kWh)'); ax.set_ylabel('sqrt(|standardized residual|)')
    ax.set_title('Scale-Location (variance-vs-mean trend)')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    scipy_stats.probplot(resid, dist='norm', plot=ax)
    ax.set_title('Normal Q-Q of residuals')
    ax.get_lines()[0].set_markerfacecolor('seagreen')
    ax.get_lines()[0].set_markeredgecolor('seagreen')
    ax.get_lines()[0].set_alpha(0.3)
    ax.get_lines()[0].set_markersize(4)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Residual diagnostics for selected model: {name} '
                f'(pooled nested-CV out-of-fold predictions, '
                f'{OUTER_CV_FOLDS}x{OUTER_CV_REPEATS} outer folds)', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'{name}_residual_diagnostics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    stats_path = os.path.join(out_dir, f'{name}_residual_diagnostics.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Residual diagnostics for selected model: {name}\n")
        f.write(f"Basis: pooled out-of-fold nested-CV predictions "
                f"({len(y_true)} points from {OUTER_CV_FOLDS}x{OUTER_CV_REPEATS} outer folds; "
                "every prediction is held-out, never seen during tuning/fitting of that fold).\n\n")
        f.write(f"Residual mean: {resid.mean():.4f} kWh (should be ~0 for an unbiased model)\n")
        f.write(f"Residual std : {resid.std():.4f} kWh\n\n")
        f.write("Breusch-Pagan test for heteroscedasticity "
                "(H0: residual variance does not depend on predicted value):\n")
        f.write(f"  LM statistic = {lm_stat:.4f}, aux R^2 = {aux_r2:.4f}, p-value = {bp_p:.4f}\n")
        if bp_p < 0.05:
            f.write("  -> p < 0.05: reject homoscedasticity. Residual variance depends on the "
                    "predicted value; report this as a limitation and interpret prediction "
                    "intervals with caution, especially at the extremes of the load range.\n")
        else:
            f.write("  -> p >= 0.05: no evidence against homoscedasticity at the 0.05 level.\n")
        shapiro_n = min(len(resid), 5000)
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(resid), size=shapiro_n, replace=False)
        sw_stat, sw_p = scipy_stats.shapiro(resid[sample_idx])
        f.write(f"\nShapiro-Wilk normality test on residuals "
                f"(n={shapiro_n}{' subsample' if shapiro_n < len(resid) else ''}): "
                f"W={sw_stat:.4f}, p={sw_p:.4f}\n")
        if sw_p < 0.05:
            f.write("  -> p < 0.05: residuals deviate from normality (see Q-Q plot).\n")
        else:
            f.write("  -> p >= 0.05: no evidence against residual normality at the 0.05 level.\n")

    print(f"[{name}] residual diagnostics saved: {plot_path}")
    print(f"[{name}] Breusch-Pagan heteroscedasticity test: LM={lm_stat:.2f}, p={bp_p:.4f} "
          f"({'HETEROSCEDASTIC' if bp_p < 0.05 else 'no evidence against homoscedasticity'})")

    return {'breusch_pagan_lm': lm_stat, 'breusch_pagan_p': bp_p,
            'shapiro_w': float(sw_stat), 'shapiro_p': float(sw_p),
            'resid_mean': float(resid.mean()), 'resid_std': float(resid.std())}


def run_nested_cv_for_model(name):
    print(f"\n################  NESTED CV : {name}  ################")
    model_dir = os.path.join(RESULTS_DIR, name)
    model_summary_dir = os.path.join(model_dir, 'summary')
    os.makedirs(model_summary_dir, exist_ok=True)

    metric_keys = ['r2', 'rmse', 'mae', 'mape']
    outer_metrics = {k: [] for k in metric_keys}
    fi_accum = None
    fi_count = 0
    fi_names = None
    best_params_rows = []
    convergence_rows = []
    convergence_curves = []
    oof_rows = []
    perm_importance_folds = []

    outer = RepeatedKFold(n_splits=OUTER_CV_FOLDS, n_repeats=OUTER_CV_REPEATS,
                          random_state=CV_RANDOM_STATE)
    X_values = X_df.reset_index(drop=True)

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X_values)):
        seed = CV_RANDOM_STATE + fold_idx

        X_tr = X_values.iloc[tr_idx]
        X_te = X_values.iloc[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        best_params, best_cv_rmse, conv = tune_on_partition(name, X_tr, y_tr, seed)
        if conv is not None:
            convergence_curves.append(conv.pop('running_best_rmse'))
            convergence_rows.append({'fold': fold_idx, 'seed': seed, **conv})

        final_model = build_tuned(name, best_params, seed)
        final_model.fit(X_tr, y_tr)
        y_pred = final_model.predict(X_te)
        ho_r2 = r2_score(y_te, y_pred)
        ho_rmse = rmse(y_te, y_pred)
        ho_mae = mean_absolute_error(y_te, y_pred)
        ho_mape = mean_absolute_percentage_error(y_te, y_pred)

        fold_perm = permutation_importance(
            final_model, X_te, y_te, n_repeats=PERM_IMPORTANCE_N_REPEATS_OUTER,
            random_state=seed, scoring='neg_root_mean_squared_error', n_jobs=-1)
        perm_importance_folds.append(fold_perm.importances_mean)

        oof_rows.append(pd.DataFrame({
            'fold': fold_idx, 'y_true': y_te, 'y_pred': y_pred,
        }))

        seed_fi = get_feature_importance(final_model)
        if seed_fi is not None:
            if fi_names is None:
                fi_names = preprocessor_feature_names(
                    final_model.named_steps['prep'])
                fi_accum = np.zeros(len(seed_fi))
            fi_accum += seed_fi
            fi_count += 1

        print(f"[{name}] fold {fold_idx:3d} | inner CV_RMSE {best_cv_rmse:.3f} | "
              f"OUTER R2 {ho_r2:.3f} RMSE {ho_rmse:.3f} MAE {ho_mae:.3f}")

        best_params_rows.append(
            {'fold': fold_idx, 'seed': seed, 'inner_cv_rmse': best_cv_rmse,
             **best_params})

        if SAVE_FOLD_FIGURES:
            fold_dir = os.path.join(model_dir, f'fold_{fold_idx:03d}')
            os.makedirs(fold_dir, exist_ok=True)

            with open(os.path.join(fold_dir, 'metrics_report.txt'), 'w') as f:
                f.write(f"=== {name} | Outer fold {fold_idx} (seed {seed}) ===\n\n")
                f.write(f"Inner Optuna CV RMSE: {best_cv_rmse:.4f}\n")
                f.write(f"Best params: {best_params}\n\n")
                f.write("Outer test: R2 {:.4f}  RMSE {:.4f}  MAE {:.4f}  MAPE {:.4f}\n"
                        .format(ho_r2, ho_rmse, ho_mae, ho_mape))

            fi_df = None
            if seed_fi is not None:
                fi_df = (pd.DataFrame({'Feature': fi_names, 'Importance': seed_fi})
                         .sort_values('Importance', ascending=False))

            plot_results('Outer test', y_te, y_pred, ho_r2, ho_rmse, ho_mae,
                         ho_mape, fi_df, 'seagreen',
                         os.path.join(fold_dir, f'outer_fold_{fold_idx:03d}.png'))

        for k, v in zip(metric_keys, [ho_r2, ho_rmse, ho_mae, ho_mape]):
            outer_metrics[k].append(v)

    n_folds = len(outer_metrics['r2'])

    conv_summary = {}
    if convergence_rows:
        conv_df = pd.DataFrame(convergence_rows)
        conv_df.to_csv(os.path.join(model_summary_dir,
                                    'optuna_convergence_per_fold.csv'), index=False)

        curves = np.vstack(convergence_curves)
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        pd.DataFrame({
            'trial': np.arange(1, len(mean_curve) + 1),
            'mean_running_best_rmse': mean_curve,
            'std_running_best_rmse': std_curve,
        }).to_csv(os.path.join(model_summary_dir, 'optuna_convergence_curve.csv'),
                  index=False)

        plt.figure(figsize=(8, 5))
        trials_x = np.arange(1, len(mean_curve) + 1)
        plt.plot(trials_x, mean_curve, color='seagreen', lw=2,
                 label='mean running-best RMSE (across outer folds)')
        plt.fill_between(trials_x, mean_curve - std_curve, mean_curve + std_curve,
                         color='seagreen', alpha=0.2, label='+/- 1 std across folds')
        plt.xlabel('Optuna trial')
        plt.ylabel('Running-best inner CV RMSE (kWh)')
        plt.title(f'{name}: Optuna convergence over {N_TRIALS} trials '
                  f'({n_folds} outer folds, {count_hyperparams(name)} tuned hyperparameters)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_summary_dir, 'optuna_convergence_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        tol_col = [c for c in conv_df.columns if c.startswith('trial_reaching_within_')][0]
        conv_summary = {
            'n_hyperparams': count_hyperparams(name),
            'mean_trial_reaching_within_1pct_of_final': float(conv_df[tol_col].mean()),
            'median_trial_reaching_within_1pct_of_final': float(conv_df[tol_col].median()),
            'pct_folds_still_improving_in_last_quarter': float(
                (conv_df['pct_improvement_in_last_quarter_of_trials'] > 5.0).mean() * 100.0),
            'mean_pct_improvement_in_last_quarter': float(
                conv_df['pct_improvement_in_last_quarter_of_trials'].mean()),
        }
        print(f"[{name}] Optuna convergence: median trial reaching within 1% of the "
              f"{N_TRIALS}-trial best = {conv_summary['median_trial_reaching_within_1pct_of_final']:.0f} "
              f"| {conv_summary['pct_folds_still_improving_in_last_quarter']:.0f}% of folds still "
              f"improving >5% in the last quarter of trials "
              f"({conv_summary['n_hyperparams']} tuned hyperparameters)")

    fi_df_mean = None
    if fi_count > 0:
        fi_mean = fi_accum / fi_count
        fi_df_mean = (pd.DataFrame({'Feature': fi_names, 'Importance': fi_mean})
                      .sort_values('Importance', ascending=False))
        fi_df_mean.to_csv(os.path.join(model_summary_dir,
                                       'feature_importance_mean.csv'), index=False)

    nice = ['R2', 'RMSE', 'MAE', 'MAPE']
    summary_rows = []
    for k, n in zip(metric_keys, nice):
        ci_lo, ci_hi = bootstrap_ci(outer_metrics[k])
        summary_rows.append({
            'Metric': n,
            'NestedCV_Mean': float(np.mean(outer_metrics[k])),
            'NestedCV_Std': float(np.std(outer_metrics[k])),
            f'CI{BOOTSTRAP_CI}_low': ci_lo,
            f'CI{BOOTSTRAP_CI}_high': ci_hi,
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(model_summary_dir, 'metrics_summary.csv'), index=False)
    pd.DataFrame({'fold': list(range(n_folds)),
                  **{f'outer_{k}': outer_metrics[k] for k in metric_keys}}
                 ).to_csv(os.path.join(model_summary_dir, 'per_fold_metrics.csv'),
                          index=False)
    pd.DataFrame(best_params_rows).to_csv(
        os.path.join(model_summary_dir, 'best_params_per_fold.csv'), index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(
        os.path.join(model_summary_dir, 'oof_predictions.csv'), index=False)

    perm_matrix = np.vstack(perm_importance_folds)
    perm_rows = []
    for j, feat in enumerate(PREDICTORS):
        col = perm_matrix[:, j]
        ci_lo, ci_hi = bootstrap_ci(col)
        perm_rows.append({
            'Feature': feat, 'PermImportance_mean': float(col.mean()),
            'PermImportance_std': float(col.std()),
            f'CI{BOOTSTRAP_CI}_low': ci_lo, f'CI{BOOTSTRAP_CI}_high': ci_hi,
        })
    perm_outer_df = pd.DataFrame(perm_rows).sort_values('PermImportance_mean', ascending=False)
    perm_outer_df.to_csv(os.path.join(model_summary_dir,
                                      'permutation_importance_outer_folds.csv'), index=False)

    plt.figure(figsize=(9, 6))
    order = perm_outer_df.sort_values('PermImportance_mean')
    xerr = np.vstack([order['PermImportance_mean'] - order[f'CI{BOOTSTRAP_CI}_low'],
                      order[f'CI{BOOTSTRAP_CI}_high'] - order['PermImportance_mean']])
    plt.barh(order['Feature'], order['PermImportance_mean'], xerr=xerr,
             capsize=3, color='steelblue')
    plt.xlabel(f'Permutation importance (increase in RMSE), {BOOTSTRAP_CI}% bootstrap CI '
              f'over {n_folds} outer folds')
    plt.title(f'{name} - Permutation Importance (held-out outer-test folds, unbiased)')
    plt.grid(alpha=0.3, axis='x'); plt.tight_layout()
    plt.savefig(os.path.join(model_summary_dir, 'permutation_importance_outer_folds.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    plot_aggregate('Nested CV (outer test)', outer_metrics, fi_df_mean, 'seagreen',
                   os.path.join(model_summary_dir, 'nested_cv_summary.png'),
                   n_folds)
    plot_ci('Nested CV (outer test)', outer_metrics,
            os.path.join(model_summary_dir, 'nested_cv_ci.png'), name)

    print(f"\n--- {name} SUMMARY (nested CV over {n_folds} outer folds, "
          f"mean +/- std [{BOOTSTRAP_CI}% bootstrap CI]) ---")
    for row in summary_rows:
        print(f"{row['Metric']:6s} | {row['NestedCV_Mean']:.4f} +/- "
              f"{row['NestedCV_Std']:.4f}  "
              f"[{row[f'CI{BOOTSTRAP_CI}_low']:.4f}, "
              f"{row[f'CI{BOOTSTRAP_CI}_high']:.4f}]")

    result = {'Model': name}
    for k, n in zip(metric_keys, nice):
        ci_lo, ci_hi = bootstrap_ci(outer_metrics[k])
        result[f'NestedCV_{n}_mean'] = float(np.mean(outer_metrics[k]))
        result[f'NestedCV_{n}_std'] = float(np.std(outer_metrics[k]))
        result[f'NestedCV_{n}_CI{BOOTSTRAP_CI}_low'] = ci_lo
        result[f'NestedCV_{n}_CI{BOOTSTRAP_CI}_high'] = ci_hi
    result.update(conv_summary)
    return result


def count_hyperparams(name):
    class _StubTrial:
        def suggest_int(self, pname, low, high, **kw):
            return low
        def suggest_float(self, pname, low, high, **kw):
            return low
        def suggest_categorical(self, pname, choices):
            return choices[0]
    return len(suggest_params(name, _StubTrial()))


def holm_bonferroni(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def paired_model_comparison(model_names, metric='rmse'):
    per_model_vals = {}
    for name in model_names:
        path = os.path.join(RESULTS_DIR, name, 'summary', 'per_fold_metrics.csv')
        if not os.path.exists(path):
            continue
        per_model_vals[name] = pd.read_csv(path)[f'outer_{metric}'].to_numpy()

    from itertools import combinations
    rows = []
    for a, b in combinations(per_model_vals.keys(), 2):
        va, vb = per_model_vals[a], per_model_vals[b]
        if len(va) != len(vb):
            continue
        diff = va - vb
        if np.allclose(diff, 0.0):
            dz, t_p, w_p = 0.0, 1.0, 1.0
        else:
            dz = float(diff.mean() / diff.std(ddof=1))
            t_p = float(scipy_stats.ttest_rel(va, vb).pvalue)
            try:
                w_p = float(scipy_stats.wilcoxon(va, vb).pvalue)
            except ValueError:
                w_p = float('nan')
        rows.append({
            'model_a': a, 'model_b': b, 'metric': metric,
            'mean_diff_a_minus_b': float(diff.mean()),
            'cohens_dz': dz, 'ttest_p': t_p, 'wilcoxon_p': w_p,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result['ttest_p_holm'] = holm_bonferroni(result['ttest_p'].to_numpy())
        result['wilcoxon_p_holm'] = holm_bonferroni(
            result['wilcoxon_p'].fillna(1.0).to_numpy())
    result.to_csv(os.path.join(SUMMARY_DIR, f'paired_model_comparison_{metric}.csv'),
                 index=False)
    return result


def select_parsimonious_winner(comp):
    n_folds = OUTER_CV_FOLDS * OUTER_CV_REPEATS
    best_row = comp.loc[comp['NestedCV_RMSE_mean'].idxmin()]
    se_best = best_row['NestedCV_RMSE_std'] / np.sqrt(n_folds)
    threshold = best_row['NestedCV_RMSE_mean'] + se_best

    within_1se = comp[comp['NestedCV_RMSE_mean'] <= threshold].copy()
    within_1se['n_hyperparams'] = within_1se['Model'].map(count_hyperparams)
    within_1se = within_1se.sort_values(['n_hyperparams', 'NestedCV_RMSE_mean'])
    parsimonious = within_1se.iloc[0]

    print(f"\n>>> Lowest nested-CV RMSE: {best_row['Model']} "
          f"({best_row['NestedCV_RMSE_mean']:.4f}, SE {se_best:.4f} over "
          f"{n_folds} outer folds)")
    print(f">>> Models within 1 SE of best (<= {threshold:.4f}): "
          f"{within_1se['Model'].tolist()}")
    print(f">>> Parsimonious (one-SE rule) winner: {parsimonious['Model']} "
          f"({int(parsimonious['n_hyperparams'])} tuned hyperparameters, "
          f"RMSE {parsimonious['NestedCV_RMSE_mean']:.4f})")
    within_1se.to_csv(os.path.join(SUMMARY_DIR, 'one_se_rule_candidates.csv'),
                      index=False)
    return best_row, parsimonious


def _model_checkpoint_paths(name):
    d = os.path.join(RESULTS_DIR, name, 'summary')
    return {
        'summary': os.path.join(d, 'metrics_summary.csv'),
        'per_fold': os.path.join(d, 'per_fold_metrics.csv'),
        'conv': os.path.join(d, 'optuna_convergence_per_fold.csv'),
    }


def load_cached_model_result(name):
    paths = _model_checkpoint_paths(name)
    if not (os.path.exists(paths['summary']) and os.path.exists(paths['per_fold'])):
        return None

    per_fold = pd.read_csv(paths['per_fold'])
    expected_folds = OUTER_CV_FOLDS * OUTER_CV_REPEATS
    if len(per_fold) != expected_folds:
        return None

    summary = pd.read_csv(paths['summary']).set_index('Metric')
    result = {'Model': name}
    for n in ['R2', 'RMSE', 'MAE', 'MAPE']:
        if n not in summary.index:
            return None
        row = summary.loc[n]
        result[f'NestedCV_{n}_mean'] = float(row['NestedCV_Mean'])
        result[f'NestedCV_{n}_std'] = float(row['NestedCV_Std'])
        result[f'NestedCV_{n}_CI{BOOTSTRAP_CI}_low'] = float(row[f'CI{BOOTSTRAP_CI}_low'])
        result[f'NestedCV_{n}_CI{BOOTSTRAP_CI}_high'] = float(row[f'CI{BOOTSTRAP_CI}_high'])

    if os.path.exists(paths['conv']):
        conv_df = pd.read_csv(paths['conv'])
        tol_cols = [c for c in conv_df.columns if c.startswith('trial_reaching_within_')]
        if tol_cols:
            tol_col = tol_cols[0]
            result.update({
                'n_hyperparams': count_hyperparams(name),
                'mean_trial_reaching_within_1pct_of_final': float(conv_df[tol_col].mean()),
                'median_trial_reaching_within_1pct_of_final': float(conv_df[tol_col].median()),
                'pct_folds_still_improving_in_last_quarter': float(
                    (conv_df['pct_improvement_in_last_quarter_of_trials'] > 5.0).mean() * 100.0),
                'mean_pct_improvement_in_last_quarter': float(
                    conv_df['pct_improvement_in_last_quarter_of_trials'].mean()),
            })
    return result


def _reset_worker_pool():
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
        print("[worker pool] reset after failure so the next model gets a fresh executor.")
    except Exception as reset_err:
        print(f"[worker pool] reset failed (non-fatal): {reset_err}")


def interpret_winner(name):
    print(f"\n################  INTERPRETATION : {name} (refit on all data)  ################")
    seed = CV_RANDOM_STATE
    best_params, best_cv_rmse, _conv = tune_on_partition(name, X_df, y_all, seed)
    print(f"[{name}] final study inner CV_RMSE {best_cv_rmse:.3f}; refitting on all data")

    pipe = build_tuned(name, best_params, seed)
    pipe.fit(X_df, y_all)
    fi_names = preprocessor_feature_names(pipe.named_steps['prep'])

    perm = permutation_importance(
        pipe, X_df, y_all, n_repeats=20, random_state=seed,
        scoring='neg_root_mean_squared_error', n_jobs=-1)
    perm_df = (pd.DataFrame({
        'Feature': PREDICTORS,
        'PermImportance_mean': perm.importances_mean,
        'PermImportance_std': perm.importances_std,
    }).sort_values('PermImportance_mean', ascending=False))
    perm_df.to_csv(os.path.join(SUMMARY_DIR, f'{name}_permutation_importance.csv'),
                   index=False)

    plt.figure(figsize=(9, 6))
    order = perm_df.sort_values('PermImportance_mean')
    plt.barh(order['Feature'], order['PermImportance_mean'],
             xerr=order['PermImportance_std'], capsize=3, color='indianred')
    plt.xlabel('Permutation importance (increase in RMSE)')
    plt.title(f'{name} - Permutation Importance (in-sample refit, EXPLANATORY ONLY -- '
             f'see permutation_importance_outer_folds.png for the unbiased estimate)')
    plt.grid(alpha=0.3, axis='x'); plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, f'{name}_permutation_importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{name}] permutation importance saved.")

    if not _HAVE_SHAP:
        print("[SHAP] package not installed -> skipping SHAP (install `shap` to enable).")
        return
    if name not in TREE_LIKE:
        print(f"[SHAP] {name} is not tree-based -> skipping TreeExplainer.")
        return
    try:
        prep = pipe.named_steps['prep']
        model = pipe.named_steps['model']
        X_trans = prep.transform(X_df)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)

        plt.figure()
        shap.summary_plot(shap_values, X_trans, feature_names=fi_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SUMMARY_DIR, f'{name}_shap_summary.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        mean_abs = np.abs(shap_values).mean(axis=0)
        pd.DataFrame({'Feature': fi_names, 'MeanAbsSHAP': mean_abs}) \
            .sort_values('MeanAbsSHAP', ascending=False) \
            .to_csv(os.path.join(SUMMARY_DIR, f'{name}_shap_importance.csv'),
                    index=False)
        print(f"[SHAP] summary saved for {name}.")
    except Exception as e:
        print(f"[SHAP] failed for {name}: {e}")


def main():
    run_t0 = time.perf_counter()
    env = capture_environment()

    all_models = list(build_model_zoo(0).keys())
    print(f"\n>>> Nested-CV of all {len(all_models)} candidate models: {all_models}")

    RESUME = os.environ.get('RESUME', '1') != '0'

    nested_rows = []
    runtime_rows = []
    for name in all_models:
        if RESUME:
            cached = load_cached_model_result(name)
            if cached is not None:
                print(f"\n################  NESTED CV : {name}  ################")
                print(f"[{name}] RESUME: complete results already exist "
                      f"({OUTER_CV_FOLDS * OUTER_CV_REPEATS} outer folds found in "
                      f"{RESULTS_DIR}\\{name}\\summary\\) -> skipping re-run.")
                nested_rows.append(cached)
                runtime_rows.append({'model': name, 'seconds': 0.0,
                                     'status': 'skipped (resumed from previous run)'})
                continue

        model_t0 = time.perf_counter()
        try:
            nested_rows.append(run_nested_cv_for_model(name))
            status = 'ok'
        except Exception as e:
            print(f"[{name}] nested CV FAILED: {e}")
            status = f'FAILED: {e}'
            _reset_worker_pool()
        model_elapsed = time.perf_counter() - model_t0
        runtime_rows.append({'model': name, 'seconds': model_elapsed, 'status': status})
        print(f"[{name}] wall-clock time: {model_elapsed:.1f}s ({model_elapsed / 60:.2f} min)")

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(os.path.join(SUMMARY_DIR, 'model_runtime_seconds.csv'), index=False)
    print("\n>>> Per-model wall-clock runtime (seconds), see summary/model_runtime_seconds.csv:")
    print(runtime_df.to_string(index=False))

    comp = pd.DataFrame(nested_rows).sort_values('NestedCV_RMSE_mean').reset_index(drop=True)
    comp.to_csv(os.path.join(SUMMARY_DIR, 'nested_cv_model_comparison.csv'), index=False)

    print("\n================ FINAL MODEL COMPARISON (tuned, nested CV) ================")
    print(comp.to_string(index=False))

    conv_cols = ['Model', 'n_hyperparams', 'median_trial_reaching_within_1pct_of_final',
                'pct_folds_still_improving_in_last_quarter',
                'mean_pct_improvement_in_last_quarter']
    conv_present = [c for c in conv_cols if c in comp.columns]
    if len(conv_present) > 1:
        conv_summary_df = comp[conv_present].sort_values(
            'n_hyperparams', ascending=False).reset_index(drop=True)
        conv_summary_df.to_csv(
            os.path.join(SUMMARY_DIR, 'optuna_convergence_summary.csv'), index=False)
        print(f"\n================ OPTUNA TRIAL-BUDGET CONVERGENCE (N_TRIALS={N_TRIALS}) "
              "================")
        print("Search-space dimensionality vs. how much of the 50-trial budget was "
              "actually needed:")
        print(conv_summary_df.to_string(index=False))
        print("(see summary/optuna_convergence_summary.csv; per-model curves in "
              "<model>/summary/optuna_convergence_curve.png)")

    paired = paired_model_comparison(all_models, metric='rmse')
    print(f"\n>>> Paired outer-fold RMSE comparisons ({len(paired)} pairs) saved to "
          f"summary/paired_model_comparison_rmse.csv")
    if not paired.empty:
        lin_pairs = paired[paired['model_a'].isin(LINEAR_FAMILY) &
                           paired['model_b'].isin(LINEAR_FAMILY)]
        if not lin_pairs.empty:
            print(">>> Linear-family pairwise comparisons "
                  "(mean_diff = RMSE_a - RMSE_b, dz = paired Cohen's d, "
                  "p-values Holm-adjusted):")
            for _, r in lin_pairs.iterrows():
                print(f"    {r['model_a']:16s} vs {r['model_b']:16s} | "
                      f"mean_diff {r['mean_diff_a_minus_b']:+.4f} kWh | "
                      f"dz {r['cohens_dz']:+.3f} | "
                      f"t-test p {r['ttest_p_holm']:.4f} | "
                      f"Wilcoxon p {r['wilcoxon_p_holm']:.4f}")

    best_row, winner = select_parsimonious_winner(comp)

    best_name = best_row['Model']
    best_fold_rmse = pd.read_csv(
        os.path.join(RESULTS_DIR, best_name, 'summary', 'per_fold_metrics.csv')
    )['outer_rmse'].to_numpy()

    diff_rows = []
    for m in comp['Model']:
        fold_rmse = pd.read_csv(
            os.path.join(RESULTS_DIR, m, 'summary', 'per_fold_metrics.csv')
        )['outer_rmse'].to_numpy()
        if m == best_name:
            diff_rows.append({'Model': m, 'mean_diff': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
            continue
        diff = fold_rmse - best_fold_rmse
        ci_lo, ci_hi = bootstrap_ci(diff)
        diff_rows.append({'Model': m, 'mean_diff': float(diff.mean()),
                          'ci_low': ci_lo, 'ci_high': ci_hi})

    diff_df = pd.DataFrame(diff_rows).sort_values('mean_diff', ascending=True).reset_index(drop=True)
    diff_df.to_csv(os.path.join(SUMMARY_DIR, 'rmse_diff_vs_best_model.csv'), index=False)

    y_pos = np.arange(len(diff_df))
    is_best = diff_df['Model'] == best_name
    xerr = np.vstack([diff_df['mean_diff'] - diff_df['ci_low'],
                      diff_df['ci_high'] - diff_df['mean_diff']])

    plt.figure(figsize=(9, max(6, 0.35 * len(diff_df))))
    plt.errorbar(diff_df.loc[~is_best, 'mean_diff'], y_pos[~is_best],
                xerr=xerr[:, ~is_best.to_numpy()], fmt='o', color='seagreen',
                ecolor='seagreen', elinewidth=1.5, capsize=4, markersize=7,
                label=f'other models ({BOOTSTRAP_CI}% paired bootstrap CI)')
    plt.scatter(diff_df.loc[is_best, 'mean_diff'], y_pos[is_best],
               marker='*', s=260, color='darkorange', zorder=5,
               edgecolor='black', linewidth=0.6,
               label=f'best model ({best_name}, reference = 0)')
    plt.axvline(0, color='gray', lw=1, ls='--', zorder=0)
    plt.yticks(y_pos, diff_df['Model'])
    plt.xlabel(f'RMSE difference from best model (kWh)\n[best = {best_name}, '
              f'{best_fold_rmse.mean():.4f} kWh mean nested-CV RMSE]')
    plt.title(f'Tuned Model Comparison, relative to best model '
              f'(nested CV, all {len(all_models)} models, paired outer folds)')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'nested_cv_model_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    with open(os.path.join(SUMMARY_DIR, 'WINNER.txt'), 'w') as f:
        f.write(f"Selected model (one-standard-error / parsimony rule): {winner['Model']}\n")
        f.write(f"  -> {int(winner['n_hyperparams'])} tuned hyperparameters "
                f"(see count_hyperparams); RMSE within 1 SE of the raw minimum.\n")
        if winner['Model'] != best_row['Model']:
            f.write(f"Raw lowest-mean-RMSE model was {best_row['Model']} "
                    f"({best_row['NestedCV_RMSE_mean']:.4f} vs selected "
                    f"{winner['NestedCV_RMSE_mean']:.4f}); the two are within "
                    "one standard error of each other, so the simpler model is "
                    "reported as the winner. See summary/one_se_rule_candidates.csv "
                    "for every model within the band and "
                    "summary/paired_model_comparison_rmse.csv for pairwise paired "
                    "t-test/Wilcoxon p-values (Holm-adjusted) and effect sizes "
                    "(Cohen's dz) across all outer-fold-matched model pairs.\n")
        f.write("\nGeneralization estimate = nested CV outer-fold scores "
                f"(mean +/- std [{BOOTSTRAP_CI}% bootstrap CI over outer folds]).\n\n")
        for m in ['RMSE', 'MAE', 'R2', 'MAPE']:
            unit = ' kWh' if m in ('RMSE', 'MAE') else ''
            f.write(
                f"Nested-CV {m:4s}: {winner[f'NestedCV_{m}_mean']:.4f} +/- "
                f"{winner[f'NestedCV_{m}_std']:.4f}{unit}  "
                f"[{BOOTSTRAP_CI}% CI {winner[f'NestedCV_{m}_CI{BOOTSTRAP_CI}_low']:.4f}, "
                f"{winner[f'NestedCV_{m}_CI{BOOTSTRAP_CI}_high']:.4f}]\n")

    oof_path = os.path.join(RESULTS_DIR, winner['Model'], 'summary', 'oof_predictions.csv')
    oof_df = pd.read_csv(oof_path)
    resid_stats = plot_residual_diagnostics(winner['Model'], oof_df, SUMMARY_DIR)
    with open(os.path.join(SUMMARY_DIR, 'WINNER.txt'), 'a') as f:
        f.write(f"\nResidual diagnostics (pooled nested-CV out-of-fold predictions, "
                f"see summary/{winner['Model']}_residual_diagnostics.png/.txt):\n")
        f.write(f"  Residual mean {resid_stats['resid_mean']:.4f} kWh, "
                f"std {resid_stats['resid_std']:.4f} kWh\n")
        f.write(f"  Breusch-Pagan heteroscedasticity test: LM={resid_stats['breusch_pagan_lm']:.2f}, "
                f"p={resid_stats['breusch_pagan_p']:.4f} "
                f"({'HETEROSCEDASTIC -- see limitations' if resid_stats['breusch_pagan_p'] < 0.05 else 'no evidence against homoscedasticity'})\n")
        f.write(f"  Shapiro-Wilk normality test: W={resid_stats['shapiro_w']:.4f}, "
                f"p={resid_stats['shapiro_p']:.4f}\n")
        top_perm = pd.read_csv(os.path.join(
            RESULTS_DIR, winner['Model'], 'summary', 'permutation_importance_outer_folds.csv'
        )).head(3)
        f.write(f"\nPermutation importance, unbiased (computed on each outer fold's held-out "
                f"test data, never seen by that fold's fitted model; see "
                f"{winner['Model']}/summary/permutation_importance_outer_folds.png/.csv for all "
                f"features with {BOOTSTRAP_CI}% bootstrap CIs across "
                f"{OUTER_CV_FOLDS * OUTER_CV_REPEATS} outer folds):\n")
        for _, r in top_perm.iterrows():
            f.write(f"  {r['Feature']:<24} {r['PermImportance_mean']:.4f} "
                    f"+/- {r['PermImportance_std']:.4f}\n")
        f.write(f"(A separate in-sample full-data-refit permutation importance is also saved "
                f"at summary/{winner['Model']}_permutation_importance.png for explanatory use "
                f"only -- it is not an unbiased estimate; prefer the outer-folds version above "
                f"for any claim about which features generalize as important.)\n")

    interpret_winner(winner['Model'])

    total_elapsed = time.perf_counter() - run_t0
    print(f"\n>>> TOTAL wall-clock runtime: {total_elapsed:.1f}s "
          f"({total_elapsed / 60:.2f} min, {total_elapsed / 3600:.2f} h)")

    with open(os.path.join(SUMMARY_DIR, 'WINNER.txt'), 'a') as f:
        f.write(f"\nTotal wall-clock runtime: {total_elapsed:.1f}s "
                f"({total_elapsed / 3600:.2f} h) across {len(all_models)} models, "
                f"{OUTER_CV_FOLDS * OUTER_CV_REPEATS} outer folds, {N_TRIALS} inner "
                f"Optuna trials/fold. See summary/model_runtime_seconds.csv for the "
                f"per-model breakdown and summary/environment.json for hardware/"
                f"library-version/seed provenance.\n")
        f.write(f"Run environment: Python {env['python_version']} on "
                f"{env['platform']} ({env['cpu_count_logical']} logical CPUs); "
                f"scikit-learn {env['package_versions']['scikit-learn']}, "
                f"optuna {env['package_versions']['optuna']}, "
                f"xgboost {env['package_versions']['xgboost']}, "
                f"lightgbm {env['package_versions']['lightgbm']}, "
                f"catboost {env['package_versions']['catboost']}.\n")

    print(f"\nDone. Results in '{RESULTS_DIR}/'.")


if __name__ == '__main__':
    _log_file = start_logging()
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file.close()

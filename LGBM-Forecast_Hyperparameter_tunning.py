# lgb_hyper.py

import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from numba import njit
# MLforecast related imports
from mlforecast import MLForecast

from mlforecast.target_transforms import Differences, GlobalSklearnTransformer, LocalStandardScaler
from mlforecast.lag_transforms import ExponentiallyWeightedMean, ExpandingMean, ExpandingStd
from sklearn.preprocessing import FunctionTransformer

# Other forecasting utilities
from window_ops.rolling import rolling_mean, rolling_std
from window_ops.shift import shift_array

# Performance optimization
from numba import njit
from functools import partial

class ForecastModelTuner:
    def __init__(self, train_data, val_data, static_features, prediction_horizon, n_trials=50):
        self.train_data = train_data
        self.val_data = val_data
        self.static_features = static_features
        self.prediction_horizon = prediction_horizon
        self.n_trials = n_trials
        self.study = None

    @staticmethod
    @njit
    def diff_over_previous(x, offset=1):
        """Computes the difference between the current value and its `offset` lag."""
        return x - np.roll(x, shift=offset)

    def objective(self, trial):
        # LightGBM parameter suggestions
        lgb_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'random_state': 23,
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 1e-3, 0.3, log=True),
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50),
            'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.5, 1.0)
        }

        lgb_model = LGBMRegressor(**lgb_params)

        @njit
        def diff_over_previous(x, offset=1):
            """Computes the difference between the current value and its `offset` lag"""
            return x - shift_array(x, offset=offset)

        sk_log1p = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        # Custom absolute function
        abs_transform = FunctionTransformer(func=lambda x: abs(x), inverse_func=lambda x: -1*x)

        fcst = MLForecast(
            models=lgb_model,
            freq='ME',
            lags=[1, 2, 3, 6, 12],
            lag_transforms={
                            1: [
                                (rolling_mean, 2),
                                (rolling_std, 2),
                                (rolling_mean, 3),
                                (rolling_std, 3),
                                (rolling_mean, 6),
                                (rolling_std, 6),
                                ExponentiallyWeightedMean(alpha=0.5),
                                ExponentiallyWeightedMean(alpha=0.8),
                                diff_over_previous, (diff_over_previous, 2), (diff_over_previous, 3)
                            ],
                            2: [
                                (rolling_mean, 2),
                                (rolling_std, 2),
                                (rolling_mean, 3),
                                (rolling_std, 3),
                                (rolling_mean, 6),
                                (rolling_std, 6),
                                ExponentiallyWeightedMean(alpha=0.5),
                                ExponentiallyWeightedMean(alpha=0.8),
                                diff_over_previous, (diff_over_previous, 2), (diff_over_previous, 3)
                            ],
                            3: [
                                (rolling_mean, 2),
                                (rolling_std, 2),
                                (rolling_mean, 3),
                                (rolling_std, 3),
                                (rolling_mean, 6),
                                (rolling_std, 6),
                                ExponentiallyWeightedMean(alpha=0.5),
                                ExponentiallyWeightedMean(alpha=0.8),
                            ],
                            6: [
                                (rolling_mean, 2),
                                (rolling_std, 2),
                                (rolling_mean, 3),
                                (rolling_std, 3),
                            ],
                            12: [
                                (rolling_mean, 2),
                                (rolling_std, 2),
                                (rolling_mean, 3),
                                (rolling_std, 3),
                            ],
                        },
            target_transforms=[LocalStandardScaler()],
            num_threads=6,
            date_features=['month', 'year'],
        )

        fcst.fit(self.train_data, static_features=self.static_features)

        valid_preds = fcst.predict(self.prediction_horizon).iloc[:, 2]

        # Zero-thresholding for small values
        valid_preds = np.where(np.abs(valid_preds) < 20, 0, valid_preds)
        validation_r2 = r2_score(self.val_data['y'], valid_preds)

        return np.abs(validation_r2)

    def tune_model(self):
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=self.n_trials)

    def get_best_results(self):
        return self.study.best_value, self.study.best_params if self.study else (None, None)

def main():
    
    final_data = pd.read_pickle('data/processed/final_data_forecast.pkl')

    # Sort final_data by unique_id and date
    final_data = final_data.sort_values(['unique_id', 'ds']).reset_index(drop=True)  # Reset index after sorting

    # Get the indices for train and validation sets
    train_idx = []
    val_idx = []
    H = 5

    for client in final_data['unique_id'].unique():
        # Get mask for current client
        client_mask = final_data['unique_id'] == client
        client_indices = final_data[client_mask].index
        
        # Last H records go to validation
        val_idx.extend(client_indices[-H:])
        # Rest goes to training
        train_idx.extend(client_indices[:-H])

    # Create train and validation sets
    train = final_data.loc[train_idx]
    val = final_data.loc[val_idx]

    STATIC = ['current_age',
                'retirement_age',
                'birth_year',
                'birth_month',
                'gender',
                'latitude',
                'longitude',
                'per_capita_income',
                'yearly_income',
                'total_debt',
                'credit_score',
                'num_credit_cards']

    train = train[['unique_id', 'ds', 'y']+STATIC]
    val = val[['unique_id', 'ds', 'y']+STATIC]

    H = 5
    n_trials = 100

    # Instantiate and run the tuner
    tuner = ForecastModelTuner(train, val, STATIC, H, n_trials)
    tuner.tune_model()

    # Retrieve and print the best results
    best_r2, best_params = tuner.get_best_results()
    print(f'Best trial R2: {best_r2}')
    print(f'Best parameters: {best_params}')

# Execute main if the script is run directly
if __name__ == "__main__":
    main()

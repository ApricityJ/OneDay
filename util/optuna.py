import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import plotly
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_param_importances


class Optuna(object):
    def __init__(self, model_type, model):
        self.model_type = model_type
        self.model = model
        self.early_stop_list = []
        self.n_trials = self.model.optuna_n_trials
        self.direction = self.model.optuna_direction

    def optimize(self) -> dict:
        print("--------- begin search params ---------")
        print(f"eval_key : {self.model.eval_key}")

        if self.model_type == "lightgbm":
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.model.magic_seed),
                                        direction=self.direction,
                                        study_name="lightgbm_search")
            study.optimize(self.lgbm_objective, n_trials=self.n_trials)
        elif self.model_type == "xgboost":
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.model.magic_seed),
                                        direction=self.direction,
                                        study_name="xgboost_search")
            study.optimize(self.xgb_objective, n_trials=self.n_trials)
        elif self.model_type == "catboost":
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.model.magic_seed),
                                        direction=self.direction,
                                        study_name="catboost_search")
            study.optimize(self.catm_objective, n_trials=self.n_trials)
        else:
            study = None
            pass

        best = study.best_params
        print(self.early_stop_list)
        best['num_boost_round'] = self.early_stop_list[study.best_trial.number]
        graph_importance = plot_param_importances(study)
        plotly.offline.plot(graph_importance)
        print("--------- done search params ---------")
        return best

    def lgbm_objective(self, trial):

        params = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                  'num_boost_round': trial.suggest_int('num_boost_round', 100, 10000, 100),
                  "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=4),
                  'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10),
                  # "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
                  # "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
                  'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10),
                  'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10),
                  "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
                  "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
                  'objective': self.model.objective,
                  'metric': self.model.metric,
                  # 'boosting': self.model.boosting,
                  'verbose': -1,
                  'seed': self.model.magic_seed,
                  'bagging_seed': self.model.magic_seed,
                  'is_unbalance': True
                  }

        if self.model.objective == 'multiclass':
            params['num_class'] = self.model.num_class

        cv_result = lgb.cv(
            params,
            self.model.lgb_train,
            num_boost_round=params['num_boost_round'],
            fobj=self.model.fobj,
            feval=self.model.feval,
            nfold=5,
            stratified=True,
            early_stopping_rounds=50,
            seed=self.model.magic_seed)

        # print(f'cv_result: {cv_result}')
        cv_score = cv_result[self.model.eval_key][-1]
        self.early_stop_list.append(len(cv_result[self.model.eval_key]))

        return cv_score

    def xgb_objective(self, trial):

        params = {"eta": trial.suggest_float("eta", 0.01, 0.5),
                  'num_boost_round': trial.suggest_int('num_boost_round', 100, 10000, 100),
                  'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                  'max_depth': trial.suggest_int('max_depth', 1, 14, 1),
                  'min_child_weight': trial.suggest_int('min_child_weight', 1, 6, 1),
                  'alpha': trial.suggest_float('reg_alpha', 0.01, 10),
                  'lambda': trial.suggest_float('reg_lambda', 0.01, 10),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.),
                  'objective': self.model.objective,
                  'eval_metric': self.model.eval_metric,
                  'verbosity': 0,
                  'seed': self.model.magic_seed
                  }

        if 'multi' in self.model.objective:
            params['num_class'] = self.model.num_class

        params['num_boost_round'] = int(params['num_boost_round'])
        cv_result = xgb.cv(
            params,
            self.model.xgb_train,
            num_boost_round=params['num_boost_round'],
            obj=self.model.obj,
            feval=self.model.feval,
            maximize=self.model.eval_maximize,
            nfold=5,
            stratified=True,
            early_stopping_rounds=50,
            seed=self.model.magic_seed)

        # print(f'cv_result: {cv_result}')
        cv_score = round(cv_result[self.model.eval_key].iloc[-1], 4)
        self.early_stop_list.append(len(cv_result[self.model.eval_key]))

        return cv_score

    def catm_objective(self, trial):

        params = {'num_boost_round': trial.suggest_int('num_boost_round', 100, 1000, 50),
                  'depth': trial.suggest_int('depth', 4, 14, 1),
                  'border_count': trial.suggest_int('border_count', 32, 255, 1),
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                  'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                  "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 1.),
                  "random_strength": trial.suggest_float("random_strength", 0.1, 10.),
                  'objective': self.model.objective,
                  'eval_metric': self.model.eval_metric,
                  'random_seed': self.model.magic_seed
                  }

        if 'multi' in self.model.objective:
            params['num_class'] = self.model.num_class

        params['num_boost_round'] = int(params['num_boost_round'])
        cv_result = cb.cv(
            params=params,
            pool=self.model.cb_train,
            fold_count=5,
            shuffle=True,
            partition_random_seed=0,
            plot=False,
            stratified=True,
            verbose=False,
            num_boost_round=params['num_boost_round'],
            early_stopping_rounds=30,
            seed=self.model.magic_seed)

        # print(f'cv_result: {cv_result}')
        cv_score = np.max(cv_result[self.model.eval_key])
        self.early_stop_list.append(np.argmax(cv_result[self.model.eval_key]) + 1)

        return cv_score

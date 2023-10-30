import numpy as np
import lightgbm as lgb
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials


class Hyperopt(object):
    def __init__(self, model_type, model):
        self.model_type = model_type
        self.model = model
        self.early_stop_dict = {}
        self.max_evals = 30

    def optimize(self) -> dict:

        print("--------- begin search params ---------")
        print(f"eval_key : {self.model.eval_key}")
        param_space = self.hyperparameter_space()
        objective = self.get_objective()
        objective.i = 0
        trials = Trials()
        best = fmin(fn=objective,
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=self.max_evals,
                    trials=trials,
                    rstate=np.random.default_rng(self.model.magic_seed))
        best['num_boost_round'] = self.early_stop_dict[trials.best_trial['tid']]
        # print(self.early_stop_dict)
        print("--------- done search params ---------")
        return best

    def get_objective(self):

        if self.model_type == "lightgbm":
            def objective(params):
                params['num_boost_round'] = int(params['num_boost_round'])
                params['num_leaves'] = int(params['num_leaves'])

                params['is_unbalance'] = True
                params['verbose'] = -1
                params['seed'] = self.model.magic_seed
                params['bagging_seed'] = self.model.magic_seed

                params['objective'] = self.model.objective
                if self.model.objective == 'multiclass':
                    params['num_class'] = self.model.num_class
                params['metric'] = self.model.metric

                cv_result = lgb.cv(
                    params,
                    self.model.lgb_train,
                    num_boost_round=params['num_boost_round'],
                    fobj=self.model.fobj,
                    feval=self.model.feval,
                    nfold=5,
                    stratified=True,
                    early_stopping_rounds=30,
                    seed=self.model.magic_seed)
                self.early_stop_dict[objective.i] = len(cv_result[self.model.eval_key])
                score = round(cv_result[self.model.eval_key][-1], 4)
                objective.i += 1
                return -score
        elif self.model_type == "xgboost":
            def objective(params):
                params['num_boost_round'] = int(params['num_boost_round'])

                params['verbosity'] = 0
                params['seed'] = self.model.magic_seed

                params['objective'] = self.model.objective
                if 'multi' in self.model.objective:
                    params['num_class'] = self.model.num_class
                params['eval_metric'] = self.model.eval_metric

                cv_result = xgb.cv(
                    params,
                    self.model.xgb_train,
                    num_boost_round=params['num_boost_round'],
                    obj=self.model.obj,
                    feval=self.model.feval,
                    maximize=self.model.eval_maximize,
                    nfold=5,
                    stratified=True,
                    early_stopping_rounds=30,
                    seed=self.model.magic_seed)
                # print(cv_result)
                self.early_stop_dict[objective.i] = len(cv_result[self.model.eval_key])
                score = round(cv_result[self.model.eval_key].iloc[-1], 4)
                objective.i += 1
                return -score
        else:
            def objective(params):
                pass
            pass

        return objective

    def hyperparameter_space(self):

        if self.model_type == "lightgbm":
            space = {
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
                'num_boost_round': hp.quniform('num_boost_round', 100, 10000, 100),  # not sure
                'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
                'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
                'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.),
                'subsample': hp.uniform('subsample', 0.5, 1.),
                'reg_alpha': hp.uniform('reg_alpha', 0.01, 10),
                'reg_lambda': hp.uniform('reg_lambda', 0.01, 10),
            }
            return space
        elif self.model_type == "xgboost":
            space = {
                'eta': hp.uniform('learning_rate', 0.01, 0.5),
                'num_boost_round': hp.quniform('num_boost_round', 100, 10000, 100),
                'gamma': hp.uniform('gamma', 0.0, 0.5),
                'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
                'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                'subsample': hp.uniform('subsample', 0.5, 1.),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
                'alpha': hp.uniform('reg_alpha', 0.01, 10),
                'lambda': hp.uniform('reg_lambda', 0.01, 10)
            }
            return space
        else:
            pass

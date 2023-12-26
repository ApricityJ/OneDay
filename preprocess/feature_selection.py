from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from constant import active_random_state
from data import loader


def select_by_boruta():
    train, _ = loader.to_df_train_test()

    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    boruta = BorutaPy(estimator=rf, n_estimators="auto", verbose=2, random_state=active_random_state)

    boruta.fit(train[:-1], train['label'])

    return boruta.transform(train)


def select_by_wrapper():
    pass

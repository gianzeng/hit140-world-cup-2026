from shared.paths import location
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.goals.regression_data import prepare

NAME = "regression_match"
FEATURES = ["total_shots", "total_yellow", "total_corners", "rank_gap"]
TARGET = "total_goals"


def split_matches(frame, groups):
    # Keep both sides of a fixture in the same split.
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1402026)
    train, test = next(split.split(frame, groups=groups))
    assert not set(groups.iloc[train]) & set(groups.iloc[test])
    X = frame[FEATURES]
    y = frame[TARGET]
    assert X.notna().all().all() and y.notna().all()
    return train, test, X, y


def fit_models(X, y, groups, train, test):
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X.iloc[train], y.iloc[train])
    pred = model.predict(X.iloc[test])
    baseline = (
        DummyRegressor(strategy="mean")
        .fit(X.iloc[train], y.iloc[train])
        .predict(X.iloc[test])
    )
    cv = -cross_val_score(
        make_pipeline(StandardScaler(), LinearRegression()),
        X.iloc[train],
        y.iloc[train],
        groups=groups.iloc[train],
        cv=GroupKFold(5),
        scoring="neg_mean_absolute_error",
    )
    return model, pred, baseline, cv


def scores(p, actual):
    return {
        "MAE": mean_absolute_error(actual, p),
        "RMSE": float(np.sqrt(mean_squared_error(actual, p))),
        "R2": r2_score(actual, p),
    }


def summarise_model(groups, train, test, y, model, pred, baseline, cv):
    result = dict(
        name=NAME,
        target=TARGET,
        features=FEATURES,
        train_rows=len(train),
        test_rows=len(test),
        train_matches=groups.iloc[train].nunique(),
        test_matches=groups.iloc[test].nunique(),
        model=scores(pred, y.iloc[test]),
        baseline=scores(baseline, y.iloc[test]),
        training_5fold_group_MAE=cv.tolist(),
        training_5fold_group_MAE_mean=float(cv.mean()),
        coefficients_per_training_SD=dict(zip(FEATURES, model[-1].coef_.tolist())),
        intercept=float(model[-1].intercept_),
        negative_predictions=int((pred < 0).sum()),
        limitation="Same-match associations. Match-disjoint split prevents opponent rows crossing folds; teams recur across folds. Generalisation to unseen teams/tournaments is untested. Count outcome may yield negative linear predictions.",
    )
    return result


def save_results(frame, train, test, y, pred, baseline, result):
    location(f"{NAME}.json", "outputs").write_text(json.dumps(result, indent=2))
    table = frame.iloc[test][["match_id"]].copy()
    table["actual"] = y.iloc[test]
    table["predicted"] = pred
    table["baseline"] = baseline
    table.to_csv(location(f"{NAME}_predictions.csv", "outputs"), index=False)
    frame.assign(
        split=np.where(np.isin(np.arange(len(frame)), train), "train", "test")
    ).to_csv(location(f"{NAME}_dataset.csv", "data"), index=False)


def run():
    frame = prepare()
    groups = frame.match_id
    train, test, X, y = split_matches(frame, groups)
    model, pred, baseline, cv = fit_models(X, y, groups, train, test)
    result = summarise_model(groups, train, test, y, model, pred, baseline, cv)
    save_results(frame, train, test, y, pred, baseline, result)
    return result


if __name__ == "__main__":
    print(run())

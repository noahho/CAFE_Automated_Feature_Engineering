import copy
import numpy as np
import pandas as pd
import tabpfn
from tabpfn.scripts.tabular_baselines import (
    gp_metric,
    knn_metric,
    xgb_metric,
    catboost_metric,
    transformer_metric,
    logistic_metric,
    autosklearn_metric,
    autosklearn2_metric,
    autogluon_metric,
    random_forest_metric,
)

from .data import get_data_split, get_X_y
from .run_llm_code import (
    run_llm_code,
    convert_categorical_to_integer_f,
    create_mappings,
)


def evaluate_dataset(
    ds, df_train, df_test, prompt_id, name, method, metric_used, max_time=30, seed=0
):

    df_train = copy.deepcopy(df_train)
    if df_test is not None:
        df_test = copy.deepcopy(df_test)

    df_train = df_train.replace([np.inf, -np.inf], np.nan)

    # Create the mappings using the train and test datasets
    mappings = create_mappings(df_train, df_test)

    # Apply the mappings to the train and test datasets
    df_train.loc[:, df_train.columns != ds[4][-1]] = df_train.loc[
        :, df_train.columns != ds[4][-1]
    ].apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )
    df_test.loc[:, df_test.columns != ds[4][-1]] = df_test.loc[
        :, df_test.columns != ds[4][-1]
    ].apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )

    df_train = df_train.astype(float)

    x, y = get_X_y(df_train, ds)

    if df_test is not None:
        # df_test = df_test.apply(lambda x: pd.factorize(x)[0])
        df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        try:
            df_test.loc[:, df_test.dtypes == object] = df_test.loc[
                :, df_test.dtypes == object
            ].apply(lambda x: pd.factorize(x)[0])
        except:
            pass
        df_test = df_test.astype(float)

        test_x, test_y = get_X_y(df_test, ds)

    if method == "autogluon":
        metric, ys, res = clf_dict[method](
            x, y, test_x, test_y, ds[3], metric_used, max_time=max_time
        )  #
    elif type(method) == str:
        metric, ys, res = clf_dict[method](
            x, y, test_x, test_y, ds[3], metric_used, max_time=max_time, no_tune={}
        )  #
    else:
        metric, ys, res = method(
            x, y, test_x, test_y, ds[3], metric_used, max_time=max_time
        )
    acc = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, ys)
    roc = tabpfn.scripts.tabular_metrics.auc_metric(test_y, ys)

    method_str = method if type(method) == str else "transformer"
    return {
        "acc": float(acc.numpy()),
        "roc": float(roc.numpy()),
        "prompt": prompt_id,
        "seed": seed,
        "name": name,
        "size": len(df_train),
        "method": method_str,
        "max_time": max_time,
        "feats": x.shape[-1],
    }


def evaluate_dataset_with_and_without_cafe(
    ds, seed, method, all_results, metric_used, prompt_id="v2"
):
    """Evaluates a dataframe with and without feature extension."""
    ds, df_train, df_test, df_train_old, df_test_old = get_data_split(ds, seed)

    f = open(f"feature_extension/{ds[0]}_{prompt_id}_0_code.txt", "r")
    code = f.read()
    f.close()

    df_train = run_llm_code(
        code, df_train, convert_categorical_to_integer=not ds[0].startswith("kaggle")
    )
    df_test = run_llm_code(
        code, df_test, convert_categorical_to_integer=not ds[0].startswith("kaggle")
    )

    method_str = method if type(method) == str else "transformer"
    all_results[f"{ds[0]}_{prompt_id}_{str(seed)}_{method_str}"] = evaluate_dataset(
        ds=ds,
        df_train=df_train,
        df_test=df_test,
        prompt_id=prompt_id,
        name=ds[0],
        method=method,
        metric_used=metric_used,
        seed=seed,
    )
    all_results[f"{ds[0]}__{str(seed)}_{method_str}"] = evaluate_dataset(
        ds=ds,
        df_train=df_train_old,
        df_test=df_test_old,
        prompt_id="",
        name=ds[0],
        method=method,
        metric_used=metric_used,
        seed=seed,
    )

    return all_results


def get_leave_one_out_importance(
    df_train, df_test, ds, method, metric_used, max_time=30
):
    """Get the importance of each feature for a dataset by dropping it in the training and prediction."""
    res_base = evaluate_dataset(
        ds,
        df_train,
        df_test,
        prompt_id="",
        name=ds[0],
        method=method,
        metric_used=metric_used,
        max_time=max_time,
    )

    importances = {}
    for feat_idx, feat in enumerate(set(df_train.columns)):
        if feat == ds[4][-1]:
            continue
        df_train_ = df_train.copy().drop(feat, axis=1)
        df_test_ = df_test.copy().drop(feat, axis=1)
        ds_ = copy.deepcopy(ds)
        # ds_[4] = list(set(ds_[4]) - set([feat]))
        # ds_[3] = list(set(ds_[3]) - set([feat_idx]))

        res = evaluate_dataset(
            ds_,
            df_train_,
            df_test_,
            prompt_id="",
            name=ds[0],
            method=method,
            metric_used=metric_used,
            max_time=max_time,
        )
        importances[feat] = (round(res_base["roc"] - res["roc"], 3),)
    return importances


clf_dict = {
    "gp": gp_metric,
    "knn": knn_metric,
    "catboost": catboost_metric,
    "xgb": xgb_metric,
    "transformer": transformer_metric,
    "logistic": logistic_metric,
    "autosklearn": autosklearn_metric,
    "autosklearn2": autosklearn2_metric,
    "autogluon": autogluon_metric,
    "random_forest": random_forest_metric,
}

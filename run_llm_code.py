import copy
import pandas as pd


def create_mappings(df_train, df_test):
    mappings = {}
    for col in df_train.columns:
        if (
            df_train[col].dtype.name == "category"
            or df_train[col].dtype.name == "object"
        ):
            combined = pd.concat([df_train[col], df_test[col]]).astype("category")
            mappings[col] = dict(enumerate(combined.cat.categories))
    return mappings


def convert_categorical_to_integer_f(column, mapping=None):
    if mapping is not None:
        return column.map(mapping).fillna(-1).astype(int)
    return column


def run_llm_code(code, df, convert_categorical_to_integer=True, fill_na=True):
    try:
        loc = {}
        glob = globals()

        df = copy.deepcopy(df)
        glob.update({"df": df})
        exec(code, glob, loc)
        df = copy.deepcopy(df)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)
        # print(code)

    return df

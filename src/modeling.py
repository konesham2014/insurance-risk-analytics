import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# =========================
# DATA PREPARATION
# =========================
def prepare_data(df, target):

    df = df.copy()

    # drop missing target
    df = df.dropna(subset=[target])

    # encode categorical columns
    categorical_cols = df.select_dtypes(include="object").columns

    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # fill missing values
    df = df.fillna(0)

    X = df.drop(columns=[target])

    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# =========================
# LINEAR REGRESSION
# =========================
def linear_regression_model(X_train, X_test, y_train, y_test):

    model = LinearRegression()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        preds
    ) ** 0.5

    r2 = r2_score(y_test, preds)

    return model, rmse, r2


# =========================
# RANDOM FOREST
# =========================
def random_forest_model(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        preds
    ) ** 0.5

    r2 = r2_score(y_test, preds)

    return model, rmse, r2


# =========================
# XGBOOST
# =========================
def xgboost_model(X_train, X_test, y_train, y_test):

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        preds
    ) ** 0.5

    r2 = r2_score(y_test, preds)

    return model, rmse, r2
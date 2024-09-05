import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def prepare():
    columns = [
        "työsuhde",
        "ikä",
        "sukupuoli",
        "kokemus",
        "koulutus",
        "kaupunki",
        "lähityö%",
        "kt",
        "rooli (norm)",
    ]

    df = pd.read_csv("data.csv", sep=";", usecols=columns)

    mapped_columns = [
        "työsuhde",
        "ikä",
        "sukupuoli",
        "koulutus",
        "kaupunki",
        "rooli (norm)",
    ]
    for column_name in mapped_columns:
        unique_values = sorted(df[column_name].dropna().unique(), key=str.lower)
        df[column_name] = df[column_name].map(
            {v: idx for idx, v in enumerate(list(unique_values))}
        )
    return df.dropna()


def fit(df):
    target = "kt"
    features = df.columns[df.columns != target]

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    slr = LinearRegression()

    slr.fit(X_train, y_train)
    return (slr, X_train, X_test, y_train, y_test)


def test(slr, X_train, X_test, y_train, y_test):
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
    x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    ax1.scatter(
        y_test_pred,
        y_test_pred - y_test,
        c="limegreen",
        marker="s",
        edgecolor="white",
        label="Test data",
    )
    ax2.scatter(
        y_train_pred,
        y_train_pred - y_train,
        c="steelblue",
        marker="o",
        edgecolor="white",
        label="Training data",
    )
    ax1.set_ylabel("Residuals")

    for ax in (ax1, ax2):
        ax.set_xlabel("Predicted values")
        ax.legend(loc="upper left")
        ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100, color="black", lw=2)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"MSE train: {mse_train:.2f}")
    print(f"MSE test: {mse_test:.2f}")
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"MAE train: {mae_train:.2f}")
    print(f"MAE test: {mae_test:.2f}")
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"R^2 train: {r2_train:.2f}")
    print(f"R^2 test: {r2_test:.2f}")

    plt.tight_layout()
    plt.show()


df = prepare()
df.to_csv("./prepared_data.csv")
print("df.shape", df.shape)
slr, X_train, X_test, y_train, y_test = fit(df)
test(slr, X_train, X_test, y_train, y_test)

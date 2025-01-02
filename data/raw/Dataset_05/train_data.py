import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return np, pd, plt, sns


@app.cell
def _(pd):
    train_file_path = "../Dataset_05/train.csv"
    train_data = pd.read_csv(train_file_path)
    return train_data, train_file_path


@app.cell
def _(train_data):
    train_data
    return


@app.cell
def _(train_data):
    train_data.head()
    return


@app.cell
def _(train_data):
    train_data.describe(include="all")
    return


@app.cell
def _(train_data):
    train_data.isnull().sum()
    return


@app.cell
def _(train_data):
    train_data.info()
    return


@app.cell
def _():
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    return LabelEncoder, label_encoder


app._unparsable_cell(
    r"""
    train_data[\"date\"] = pd.to_datetime(train_data[\"date\"])\"]
    train_data[\"date\"] = train_data[\"date\"].astype(\"int64\")
    """,
    name="_"
)


@app.cell
def _(label_encoder, train_data):
    categorial_cols = ["country", "store", "product"]
    for col in categorial_cols:
        train_data[col] = label_encoder.fit_transform(train_data[col])
    return categorial_cols, col


@app.cell
def _(plt, sns, train_data):
    corr_matrix = train_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()
    return (corr_matrix,)


if __name__ == "__main__":
    app.run()

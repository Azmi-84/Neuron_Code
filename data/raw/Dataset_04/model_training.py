import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import seaborn as sns
    import matplotlib.pyplot as plt
    import missingno as msno
    return LabelEncoder, StandardScaler, msno, np, pd, plt, sns


@app.cell
def _(pd):
    data = pd.read_csv("../Dataset_04/train_cleaned.csv")
    return (data,)


@app.cell
def _(data):
    data.drop(columns=["id"])
    return


@app.cell
def _():
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import CategoryEmbeddingModelConfig
    from pytorch_tabular.config import (
        DataConfig,
        OptimizerConfig,
        TrainerConfig,
        ExperimentConfig,
    )
    return (
        CategoryEmbeddingModelConfig,
        DataConfig,
        ExperimentConfig,
        OptimizerConfig,
        TabularModel,
        TrainerConfig,
    )


@app.cell
def _():
    numeric_cols = [
        "Age",
        "Annual Income",
        "Number of Dependents",
        "Health Score",
        "Previous Claims",
        "Vehicle Age",
        "Credit Score",
        "Insurance Duration",
    ]
    categorical_cols = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Occupation",
        "Location",
        "Policy Type",
        "Smoking Status",
        "Exercise Frequency",
        "Property Type",
    ]
    return categorical_cols, numeric_cols


@app.cell
def _(
    CategoryEmbeddingModelConfig,
    DataConfig,
    OptimizerConfig,
    TabularModel,
    TrainerConfig,
    categorical_cols,
    data,
    numeric_cols,
):
    data_config = DataConfig(
        target=["Premium Amount"],
        categorical_cols=categorical_cols,
        continuous_cols=numeric_cols,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=True,
        max_epochs=100,
        batch_size=1024,
        progress_bar=True,
    )

    optimizer_config = OptimizerConfig()

    model_config = CategoryEmbeddingModelConfig(
        task="regression",
        layers="1024-512-256-128",
        activation="LeakyReLU",
        learning_rate=1e-3,
        embedding_dropout=0.1,
        dropout=0.2,
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    tabular_model.fit(train=data)

    tabular_model.save_model("../Dataset_04/model/")
    return (
        data_config,
        model_config,
        optimizer_config,
        tabular_model,
        trainer_config,
    )


@app.cell
def _(pd):
    prediction_dataset = pd.read_csv("../Dataset_04/test_cleaned.csv")
    return (prediction_dataset,)


@app.cell
def _(prediction_dataset):
    prediction_dataset.describe(include="all")
    return


@app.cell
def _(categorical_cols, prediction_dataset):
    for col in categorical_cols:
        if col in prediction_dataset.columns:
            prediction_dataset[col] = prediction_dataset[col].astype("category")
        else:
            raise ValueError(f"Missing categorical column: {col}")
    return (col,)


@app.cell
def _(numeric_cols, prediction_dataset):
    for cols in numeric_cols:
        if cols in prediction_dataset.columns:
            prediction_dataset[cols] = prediction_dataset[cols].astype("float64")
        else:
            raise ValueError(f"Missing numeric column: {cols}")
    return (cols,)


@app.cell
def _(TabularModel):
    print(dir(TabularModel))
    return


@app.cell
def _(TabularModel):
    trained_model = TabularModel.load_model("../Dataset_04/model/")
    return (trained_model,)


@app.cell
def _(prediction_dataset, trained_model):
    predictions = trained_model.predict(prediction_dataset)
    return (predictions,)


@app.cell
def _(predictions):
    print(predictions)
    print(predictions.dtypes)  # Column data types
    print(predictions.columns)  # Column names
    return


@app.cell
def _(np, pd, predictions):
    predictions_flattened = predictions[
        "Premium Amount_prediction"
    ].values.flatten()

    ids = np.arange(1200000, 1200000 + len(predictions_flattened))

    submission_df = pd.DataFrame(
        {"id": ids, "Premium Amount": predictions_flattened}
    )

    submission_df.to_csv("submission.csv", index=False)

    print("Submission file saved as submission.csv")
    return ids, predictions_flattened, submission_df


if __name__ == "__main__":
    app.run()

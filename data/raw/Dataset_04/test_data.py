import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full", auto_download=["ipynb"])


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
    file_path = "../Dataset_04/test.csv"
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    return data, file_path


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    data.describe(include="all")
    return


@app.cell
def _(data, plt, sns):
    missing_mask = data.isnull()
    plt.figure(figsize=(10, 5))
    sns.heatmap(missing_mask, cbar=False, cmap="viridis")
    plt.title("Missing values in the dataset")
    plt.show()
    return (missing_mask,)


@app.cell
def _(data):
    for missing in data.columns:
        missing_values = data[missing].isnull().sum()
        if missing_values > 0:
            print(f"Column {missing} has {missing_values} missing values.")
    return missing, missing_values


@app.cell
def _(LabelEncoder, data):
    label_encoder = LabelEncoder()
    categorical_columns = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Occupation",
        "Location",
        "Policy Type",
        "Customer Feedback",
        "Smoking Status",
        "Exercise Frequency",
        "Property Type",
    ]
    for column in categorical_columns:
        if column in data.columns:
            data[column] = label_encoder.fit_transform(data[column])
    return categorical_columns, column, label_encoder


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data, pd):
    if "Policy Start Date" in data.columns:
        data["Policy Start Date"] = pd.to_datetime(data["Policy Start Date"])
        data["Policy Start Date"] = data["Policy Start Date"].astype("int64")
    else:
        print("Column 'Policy Start Date' does not exist.")

    correlation_matrix = data.corr()
    correlation_matrix
    return (correlation_matrix,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (
        RandomForestRegressor,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        train_test_split,
    )


@app.cell
def _(correlation_matrix):
    credit_score_target_column = "Credit Score"
    credit_score_threshold = 0.0005

    credit_score_correlation = correlation_matrix[credit_score_target_column]

    credit_score_predictor_columns = [
        feature
        for feature in credit_score_correlation.index
        if abs(credit_score_correlation[feature]) > credit_score_threshold
    ]
    credit_score_predictor_columns
    return (
        credit_score_correlation,
        credit_score_predictor_columns,
        credit_score_target_column,
        credit_score_threshold,
    )


@app.cell
def _(
    RandomForestRegressor,
    credit_score_predictor_columns,
    credit_score_target_column,
    data,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    train_test_split,
):
    training_data_credit = data[data[credit_score_target_column].notnull()]
    missing_data_credit = data[data[credit_score_target_column].isnull()]

    x_full_credit = training_data_credit[credit_score_predictor_columns]
    y_full_credit = training_data_credit[credit_score_target_column]

    x_train_credit, x_val_credit, y_train_credit, y_val_credit = train_test_split(
        x_full_credit, y_full_credit, test_size=0.2, random_state=42
    )

    credit_score_model = RandomForestRegressor(random_state=42, n_estimators=100)
    credit_score_model.fit(x_train_credit, y_train_credit)

    y_val_credit_predictions = credit_score_model.predict(x_val_credit)

    mae_credit = mean_absolute_error(y_val_credit, y_val_credit_predictions)
    mse_credit = mean_squared_error(y_val_credit, y_val_credit_predictions)
    r2_credit = r2_score(y_val_credit, y_val_credit_predictions)

    print(f"Mean Absolute Error (MAE): {mae_credit}")
    print(f"Mean Squared Error (MSE): {mse_credit}")
    print(f"R-squared (R^2): {r2_credit}")
    return (
        credit_score_model,
        mae_credit,
        missing_data_credit,
        mse_credit,
        r2_credit,
        training_data_credit,
        x_full_credit,
        x_train_credit,
        x_val_credit,
        y_full_credit,
        y_train_credit,
        y_val_credit,
        y_val_credit_predictions,
    )


@app.cell
def _(y_val_credit_predictions):
    for q in y_val_credit_predictions:
        print(f"Predicted value: {q}")
    return (q,)


@app.cell
def _(data):
    for r in data["Credit Score"]:
        print(f"Actual value: {r}")
    return (r,)


@app.cell
def _(
    credit_score_model,
    credit_score_predictor_columns,
    credit_score_target_column,
    data,
    missing_data_credit,
):
    missing_data_preds_credit = credit_score_model.predict(
        missing_data_credit[credit_score_predictor_columns]
    )
    data.loc[
        data[credit_score_target_column].isnull(), credit_score_target_column
    ] = missing_data_preds_credit
    return (missing_data_preds_credit,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(correlation_matrix):
    age_target_column = "Age"
    age_threshold = 0.0003

    age_correlation = correlation_matrix[age_target_column]

    age_predictor_columns = [
        feature
        for feature in age_correlation.index
        if abs(age_correlation[feature]) > age_threshold
    ]
    age_predictor_columns
    return (
        age_correlation,
        age_predictor_columns,
        age_target_column,
        age_threshold,
    )


@app.cell
def _(
    RandomForestRegressor,
    age_predictor_columns,
    age_target_column,
    data,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    train_test_split,
):
    training_data_age = data[data[age_target_column].notnull()]
    missing_data_age = data[data[age_target_column].isnull()]

    x_full_age = training_data_age[age_predictor_columns]
    y_full_age = training_data_age[age_target_column]

    x_train_age, x_val_age, y_train_age, y_val_age = train_test_split(
        x_full_age, y_full_age, test_size=0.2, random_state=42
    )

    age_model = RandomForestRegressor(random_state=42, n_estimators=100)
    age_model.fit(x_train_age, y_train_age)

    y_val_age_predictions = age_model.predict(x_val_age)

    mae_age = mean_absolute_error(y_val_age, y_val_age_predictions)
    mse_age = mean_squared_error(y_val_age, y_val_age_predictions)
    r2_age = r2_score(y_val_age, y_val_age_predictions)

    print(f"Mean Absolute Error (MAE): {mae_age}")
    print(f"Mean Squared Error (MSE): {mse_age}")
    print(f"R-squared (R^2): {r2_age}")
    return (
        age_model,
        mae_age,
        missing_data_age,
        mse_age,
        r2_age,
        training_data_age,
        x_full_age,
        x_train_age,
        x_val_age,
        y_full_age,
        y_train_age,
        y_val_age,
        y_val_age_predictions,
    )


@app.cell
def _(y_val_age_predictions):
    for s in y_val_age_predictions:
        print(f"Predicted value: {s}")
    return (s,)


@app.cell
def _(data):
    for t in data["Age"]:
        print(f"Actual value: {t}")
    return (t,)


@app.cell
def _(
    age_model,
    age_predictor_columns,
    age_target_column,
    data,
    missing_data_age,
):
    missing_data_preds_age = age_model.predict(
        missing_data_age[age_predictor_columns]
    )
    data.loc[data[age_target_column].isnull(), age_target_column] = (
        missing_data_preds_age
    )
    return (missing_data_preds_age,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(correlation_matrix):
    target_column = "Annual Income"
    threshold = 0.0005
    target_correlation = correlation_matrix[target_column]
    predictors = [
        feature
        for feature in target_correlation.index
        if abs(target_correlation[feature]) > threshold
    ]
    predictors
    return predictors, target_column, target_correlation, threshold


@app.cell
def _(
    RandomForestRegressor,
    data,
    mean_absolute_error,
    mean_squared_error,
    predictors,
    r2_score,
    target_column,
    train_test_split,
):
    train_data = data[data[target_column].notnull()]
    missing_data = data[data[target_column].isnull()]

    x_train = train_data[predictors]
    y_train = train_data[target_column]

    x_train_set, x_val, y_train_set, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(x_train_set, y_train_set)

    y_pred = model.predict(x_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R^2): {r2}")
    return (
        mae,
        missing_data,
        model,
        mse,
        r2,
        train_data,
        x_train,
        x_train_set,
        x_val,
        y_pred,
        y_train,
        y_train_set,
        y_val,
    )


@app.cell
def _(data, missing_data, model, predictors, target_column):
    missing_data_preds = model.predict(missing_data[predictors])
    data.loc[data[target_column].isnull(), target_column] = missing_data_preds
    return (missing_data_preds,)


@app.cell
def _(y_pred):
    for u in y_pred:
        print(f"Predicted value: {u}")
    return (u,)


@app.cell
def _(data):
    for v in data["Annual Income"]:
        print(f"Actual value: {v}")
    return (v,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(correlation_matrix):
    health_score_target_column = "Health Score"
    health_score_threshold = 0.0005

    health_score_correlation = correlation_matrix[health_score_target_column]

    health_score_predictor_columns = [
        feature
        for feature in health_score_correlation.index
        if abs(health_score_correlation[feature]) > health_score_threshold
    ]

    health_score_predictor_columns
    return (
        health_score_correlation,
        health_score_predictor_columns,
        health_score_target_column,
        health_score_threshold,
    )


@app.cell
def _(
    RandomForestRegressor,
    data,
    health_score_predictor_columns,
    health_score_target_column,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    train_test_split,
):
    training_data_health = data[data[health_score_target_column].notnull()]
    missing_data_health = data[data[health_score_target_column].isnull()]

    X_full_health = training_data_health[health_score_predictor_columns]
    y_full_health = training_data_health[health_score_target_column]

    X_train_health, X_val_health, y_train_health, y_val_health = train_test_split(
        X_full_health, y_full_health, test_size=0.2, random_state=42
    )

    health_score_model = RandomForestRegressor(random_state=42, n_estimators=100)
    health_score_model.fit(X_train_health, y_train_health)

    y_val_health_predictions = health_score_model.predict(X_val_health)

    mae_health = mean_absolute_error(y_val_health, y_val_health_predictions)
    mse_health = mean_squared_error(y_val_health, y_val_health_predictions)
    r2_health = r2_score(y_val_health, y_val_health_predictions)

    print(f"Mean Absolute Error (MAE): {mae_health}")
    print(f"Mean Squared Error (MSE): {mse_health}")
    print(f"R-squared (R^2): {r2_health}")
    return (
        X_full_health,
        X_train_health,
        X_val_health,
        health_score_model,
        mae_health,
        missing_data_health,
        mse_health,
        r2_health,
        training_data_health,
        y_full_health,
        y_train_health,
        y_val_health,
        y_val_health_predictions,
    )


@app.cell
def _(y_val_health_predictions):
    for m in y_val_health_predictions:
        print(f"Predicted value: {m}")
    return (m,)


@app.cell
def _(data):
    for n in data["Health Score"]:
        print(f"Actual value: {n}")
    return (n,)


@app.cell
def _(
    data,
    health_score_model,
    health_score_predictor_columns,
    health_score_target_column,
    missing_data_health,
):
    missing_data_preds_health = health_score_model.predict(
        missing_data_health[health_score_predictor_columns]
    )
    data.loc[
        data[health_score_target_column].isnull(), health_score_target_column
    ] = missing_data_preds_health
    return (missing_data_preds_health,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(correlation_matrix):
    dependents_target_column = "Number of Dependents"
    dependents_threshold = 0.0005

    dependents_correlation = correlation_matrix[dependents_target_column]

    dependents_predictor_columns = [
        feature
        for feature in dependents_correlation.index
        if abs(dependents_correlation[feature]) > dependents_threshold
    ]

    dependents_predictor_columns
    return (
        dependents_correlation,
        dependents_predictor_columns,
        dependents_target_column,
        dependents_threshold,
    )


@app.cell
def _(
    RandomForestRegressor,
    data,
    dependents_predictor_columns,
    dependents_target_column,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    train_test_split,
):
    training_data_dependents = data[data[dependents_target_column].notnull()]
    missing_data_dependents = data[data[dependents_target_column].isnull()]

    X_full_dependents = training_data_dependents[dependents_predictor_columns]
    y_full_dependents = training_data_dependents[dependents_target_column]

    X_train_dependents, X_val_dependents, y_train_dependents, y_val_dependents = (
        train_test_split(
            X_full_dependents, y_full_dependents, test_size=0.2, random_state=42
        )
    )

    dependents_score_model = RandomForestRegressor(
        random_state=42, n_estimators=100
    )
    dependents_score_model.fit(X_train_dependents, y_train_dependents)

    y_val_dependents_predictions = dependents_score_model.predict(X_val_dependents)

    mae_dependents = mean_absolute_error(
        y_val_dependents, y_val_dependents_predictions
    )
    mse_dependents = mean_squared_error(
        y_val_dependents, y_val_dependents_predictions
    )
    r2_dependents = r2_score(y_val_dependents, y_val_dependents_predictions)

    print(f"Mean Absolute Error (MAE): {mae_dependents}")
    print(f"Mean Squared Error (MSE): {mse_dependents}")
    print(f"R-squared (R^2): {r2_dependents}")
    return (
        X_full_dependents,
        X_train_dependents,
        X_val_dependents,
        dependents_score_model,
        mae_dependents,
        missing_data_dependents,
        mse_dependents,
        r2_dependents,
        training_data_dependents,
        y_full_dependents,
        y_train_dependents,
        y_val_dependents,
        y_val_dependents_predictions,
    )


@app.cell
def _(y_val_dependents_predictions):
    for a in y_val_dependents_predictions:
        print(f"Predicted value: {a}")
    return (a,)


@app.cell
def _(data):
    for b in data["Number of Dependents"]:
        print(f"Actual value: {b}")
    return (b,)


@app.cell
def _(
    claims_score_model,
    data,
    missing_data_claims,
    previous_claims_predictor_columns,
    previous_claims_target_column,
):
    missing_data_preds_claims = claims_score_model.predict(
        missing_data_claims[previous_claims_predictor_columns]
    )
    data.loc[
        data[previous_claims_target_column].isnull(), previous_claims_target_column
    ] = missing_data_preds_claims
    return (missing_data_preds_claims,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(correlation_matrix, data, train_test_split):
    previous_claims_target_column = "Previous Claims"
    previous_claims_threshold = 0.0005

    previous_claims_correlation = correlation_matrix[previous_claims_target_column]

    previous_claims_predictor_columns = [
        feature
        for feature in previous_claims_correlation.index
        if abs(previous_claims_correlation[feature]) > previous_claims_threshold
    ]

    training_data_claims = data[data[previous_claims_target_column].notnull()]
    missing_data_claims = data[data[previous_claims_target_column].isnull()]

    X_full_claims = training_data_claims[previous_claims_predictor_columns]
    y_full_claims = training_data_claims[previous_claims_target_column]

    X_train_claims, X_val_claims, y_train_claims, y_val_claims = train_test_split(
        X_full_claims, y_full_claims, test_size=0.2, random_state=42
    )
    return (
        X_full_claims,
        X_train_claims,
        X_val_claims,
        missing_data_claims,
        previous_claims_correlation,
        previous_claims_predictor_columns,
        previous_claims_target_column,
        previous_claims_threshold,
        training_data_claims,
        y_full_claims,
        y_train_claims,
        y_val_claims,
    )


@app.cell
def _(
    RandomForestRegressor,
    X_train_claims,
    X_val_claims,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    y_train_claims,
    y_val_claims,
):
    claims_score_model = RandomForestRegressor(random_state=42, n_estimators=100)
    claims_score_model.fit(X_train_claims, y_train_claims)

    y_val_claims_predictions = claims_score_model.predict(X_val_claims)

    mae_claims = mean_absolute_error(y_val_claims, y_val_claims_predictions)
    mse_claims = mean_squared_error(y_val_claims, y_val_claims_predictions)
    r2_claims = r2_score(y_val_claims, y_val_claims_predictions)

    print(f"Mean Absolute Error (MAE): {mae_claims}")
    print(f"Mean Squared Error (MSE): {mse_claims}")
    print(f"R-squared (R^2): {r2_claims}")
    return (
        claims_score_model,
        mae_claims,
        mse_claims,
        r2_claims,
        y_val_claims_predictions,
    )


@app.cell
def _(y_val_claims_predictions):
    for c in y_val_claims_predictions:
        print(f"Predicted value: {c}")
    return (c,)


@app.cell
def _(data):
    for d in data["Previous Claims"]:
        print(f"Actual value: {d}")
    return (d,)


@app.cell
def _(
    data,
    dependents_predictor_columns,
    dependents_score_model,
    dependents_target_column,
    missing_data_dependents,
):
    missing_data_preds_dependents = dependents_score_model.predict(
        missing_data_dependents[dependents_predictor_columns]
    )
    data.loc[data[dependents_target_column].isnull(), dependents_target_column] = (
        missing_data_preds_dependents
    )
    return (missing_data_preds_dependents,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(data):
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=3)
    data["Vehicle Age"] = imputer.fit_transform(data[["Vehicle Age"]])
    return KNNImputer, imputer


@app.cell
def _(KNNImputer, data):
    imputer_insurance = KNNImputer(n_neighbors=3)
    data["Insurance Duration"] = imputer_insurance.fit_transform(
        data[["Insurance Duration"]]
    )
    return (imputer_insurance,)


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(data):
    data.to_csv("../Dataset_04/test_cleaned.csv", index=False)
    return


@app.cell
def _(pd):
    cleand_data = pd.read_csv("../Dataset_04/test_cleaned.csv")
    for e in cleand_data.columns:
        print(f"Column '{e}' has {cleand_data[e].isnull().sum()} missing values")
    return cleand_data, e


if __name__ == "__main__":
    app.run()

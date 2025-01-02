import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import requests
    return (requests,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    file_path = "../Dataset_03/Depression_Student_Dataset.csv"
    return (file_path,)


@app.cell
def _(file_path, pd):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    return (data,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    data.describe(include="all")
    return


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data):
    data.isnull().sum()
    return


@app.cell
def _(data):
    for col in data.columns:
        print(f"{col}: Min = {data[col].min()}, Max = {data[col].max()}")
    return (col,)


@app.cell
def _():
    from sklearn.preprocessing import LabelEncoder
    return (LabelEncoder,)


@app.cell
def _(LabelEncoder):
    encoder = LabelEncoder()
    return (encoder,)


@app.cell
def _(data, encoder):
    mappings = {}

    columns_to_encode = [
        "Gender",
        "Age",
        "Academic Pressure",
        "Study Satisfaction",
        "Sleep Duration",
        "Dietary Habits",
        "Suicidal Thoughts",
        "Study Hours",
        "Financial Stress",
        "Family History of Mental Illness",
        "Depression",
    ]

    for column in columns_to_encode:
        data[column] = encoder.fit_transform(data[column])
        mappings[column] = {
            original: encoded
            for original, encoded in zip(
                encoder.classes_, encoder.transform(encoder.classes_)
            )
        }
        print(f"Mapping for {column}: {mappings[column]}")

    print(data.head())
    return column, columns_to_encode, mappings


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell
def _(data, plt, sns):
    print("\nGenerating Correlation Matrix...")
    correlation_matrix = data.corr()

    num_features = len(correlation_matrix.columns)
    fig_width = max(8, num_features)
    fig_height = max(6, num_features * 0.5)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    return correlation_matrix, fig_height, fig_width, num_features


@app.cell
def _(correlation_matrix):
    threshold = 0.2

    relevant_features = correlation_matrix['Depression'][
        correlation_matrix['Depression'].abs() > threshold
    ].index.tolist()

    print("Relevant features based on correlation:", relevant_features)
    return relevant_features, threshold


@app.cell
def _(data, relevant_features):
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = data[relevant_features].drop(columns=["Depression"])
    y = data["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)

    rfe = RFE(estimator=model, n_features_to_select=5)
    rfe.fit(X_train, y_train)

    selected_features = X.columns[rfe.support_].tolist()
    print("Selected features by RFE:", selected_features)
    return (
        RFE,
        RandomForestClassifier,
        X,
        X_test,
        X_train,
        model,
        rfe,
        selected_features,
        train_test_split,
        y,
        y_test,
        y_train,
    )


@app.cell
def _(X, X_train, model, pd, plt, sns, y_train):
    model.fit(X_train, y_train)

    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("Feature importance:\n", feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()
    return (feature_importance,)


@app.cell
def _(X, selected_features):
    final_features = selected_features
    X_final = X[final_features]
    print("Final dataset shape:", X_final.shape)
    return X_final, final_features


@app.cell
def _(data):
    print(data['Depression'].value_counts())
    return


@app.cell
def _(X_test, X_train, y_test, y_train):
    import xgboost as xgb
    from sklearn.metrics import classification_report, accuracy_score

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return accuracy_score, classification_report, xgb, xgb_model, y_pred


@app.cell
def _(plt, xgb, xgb_model):
    xgb.plot_importance(xgb_model)
    plt.show()
    return


@app.cell
def _(model):
    import joblib
    joblib.dump(model, "xgboost_model.pkl")
    return (joblib,)


@app.cell
def _(joblib):
    final_model = joblib.load("xgboost_model.pkl")
    return (final_model,)


if __name__ == "__main__":
    app.run()

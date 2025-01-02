import marimo

app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return pd, plt, sns


@app.cell
def _(pd):
    file_path = "/home/abdullahalazmi/Downloads/predictive_maintenance/data/raw/Dataset_02/MetroPT3(AirCompressor).csv"
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    return data


@app.cell
def _(data):
    print("\nDataset Summary:")
    print(data.describe(include="all"))
    print("\nDataset Info:")
    print(data.info())
    return


@app.cell
def _(data):
    if "timestamp" in data.columns:
        try:
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            print(
                "\nTimestamp Range:",
                data["timestamp"].min(),
                "-",
                data["timestamp"].max(),
            )
            data.drop("timestamp", axis=1, inplace=True)
        except Exception as e:
            print("Error processing timestamp column:", e)
    return data


@app.cell
def _(data):
    print("\nMissing Values:")
    missing_values = data.isnull().sum()
    print(missing_values)
    if missing_values.any():
        print("\nFilling missing values with column means...")
        data.fillna(data.mean(), inplace=True)
    return data


@app.cell
def _(data):
    print("\nData Types:")
    print(data.dtypes)
    print("\nUnique Values:")
    print(data.nunique())
    return


@app.cell
def _(data, plt, sns):
    print("\nGenerating Correlation Matrix...")
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    return correlation_matrix


@app.cell
def _(data):
    target_column = "LPS"
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    y = data[target_column]
    X = data.drop(columns=[target_column, "Unnamed: 0"], errors="ignore")
    return X, y


@app.cell
def _(X, y):
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    print("\nSplitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nHandling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled Training Set Shape: {X_train_resampled.shape}")

    return X_train_resampled, X_test, y_train_resampled, y_test


@app.cell
def _(X_train_resampled, y_train_resampled, X):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE

    print("\nPerforming Feature Selection...")
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=7)
    rfe.fit(X_train_resampled, y_train_resampled)

    selected_features = X.columns[rfe.support_]
    print("\nSelected Features:", list(selected_features))
    return selected_features, rfe


@app.cell
def _(X_train_resampled, X_test, selected_features):
    X_train_resampled = X_train_resampled[selected_features]
    X_test = X_test[selected_features]
    return X_train_resampled, X_test


@app.cell
def _(X_train_resampled, y_train_resampled, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    print("\nTraining the Random Forest Model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model, y_pred


@app.cell
def _(model, selected_features):
    print("\nFeature Importance:")
    importance = model.feature_importances_
    for feature, imp in zip(selected_features, importance):
        print(f"Feature: {feature}, Importance: {imp:.3f}")
    return importance


if __name__ == "__main__":
    app.run()

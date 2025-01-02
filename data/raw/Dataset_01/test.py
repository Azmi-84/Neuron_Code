    import joblib
    import numpy as np
    import pandas as pd

    # Load the trained model and encoders
    model = joblib.load(
        "/home/abdullahalazmi/Downloads/predictive_maintenance/data/raw/multi_output_rf_model.pkl"
    )
    type_encoder = joblib.load(
        "/home/abdullahalazmi/Downloads/predictive_maintenance/data/raw/type_encoder.pkl"
    )
    failure_encoder = joblib.load(
        "/home/abdullahalazmi/Downloads/predictive_maintenance/data/raw/failure_encoder.pkl"
    )

    # Prepare new data
    new_data = {
        "Type": "H",
        "Air temperature [K]": 3000,
        "Process temperature [K]": 310000,
        "Rotational speed [rpm]": 120000,
        "Torque [Nm]": 0.50,
        "Tool wear [min]": 0.200,
    }

    # Create a DataFrame with a single row
    df = pd.DataFrame([new_data])

    # Apply transformations
    df["Type"] = type_encoder.transform(df[["Type"]])
    df["Rotational speed [rpm]"] = np.log1p(df["Rotational speed [rpm]"])
    df["Torque [Nm]"] = np.log1p(df["Torque [Nm]"])

    # Ensure correct feature order
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    input_features = df[feature_columns]

    # Make predictions
    predictions = model.predict(input_features)
    predicted_failure_type = failure_encoder.inverse_transform([predictions[0, 1]])

    print("\nPrediction Results:")
    print("-" * 50)
    print(f"Machine Status: {'Failed' if predictions[0, 0] == 1 else 'Normal'}")
    print(f"Failure Type: {predicted_failure_type[0]}")
    print("-" * 50)

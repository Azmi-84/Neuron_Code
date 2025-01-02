import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Read the CSV file
df = pd.read_csv("predictive_maintenance.csv")

# Initialize the encoders
type_encoder = LabelEncoder()
failure_encoder = LabelEncoder()

# Fit the encoders
type_encoder.fit(df["Type"])
failure_encoder.fit(df["Failure Type"])

# Print unique values for verification
print("Unique values in Type column:", df["Type"].unique())
print("Encoded values for Type:", type_encoder.transform(df["Type"].unique()))
print("\nUnique values in Failure Type column:", df["Failure Type"].unique())
print(
    "Encoded values for Failure Type:",
    failure_encoder.transform(df["Failure Type"].unique()),
)

# Save the encoders
joblib.dump(type_encoder, "type_encoder.pkl")
joblib.dump(failure_encoder, "failure_encoder.pkl")

print("\nEncoders have been saved as type_encoder.pkl and failure_encoder.pkl")

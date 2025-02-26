
# python check_StandardScaler_pkl.py

import numpy as np
import joblib

# Step 1: Load the StandardScaler
scaler = joblib.load("scaler.pkl")

# Step 2: Create sample data (similar to training data)
sample_data = np.array([[1000, 200, 300], [1500, 500, 600]])

# Step 3: Apply scaling
scaled_data = scaler.transform(sample_data)

# Print the scaled data
print("Scaled Data:")
print(scaled_data)

# Step 4: Verify results by applying inverse_transform (optional)
original_data = scaler.inverse_transform(scaled_data)

print("\nReversed Data (Original Form):")
print(original_data)
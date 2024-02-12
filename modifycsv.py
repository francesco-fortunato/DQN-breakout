import pandas as pd

# Load the CSV file
csv_filename = "training_stats.csv"  # Replace with the actual filename
df = pd.read_csv(csv_filename)

# Set the maximum value for "Model Updates"
max_model_updates = 100000

# Update values greater than the maximum
df['Model Updates'] = df['Model Updates'].apply(lambda x: min(x, max_model_updates))

# Save the modified DataFrame back to a CSV file
modified_csv_filename = "modified_" + csv_filename
df.to_csv(modified_csv_filename, index=False)

# Display the modified DataFrame
print(df)

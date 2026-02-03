import pandas as pd

# Load the CSV file
file_path ="training_data.csv"
file_path1 = "classified_results.csv"  # Replace with your actual file path
classified_results = pd.read_csv(file_path1)
training_data= pd.read_csv(file_path)

# Count the occurrences of each label
label_counts = training_data['label'].value_counts()

# Check for balance
print("Label distribution:")
print(label_counts)

if label_counts[0] == label_counts[1]:
   print("The dataset has an equal number of human and machine-generated texts.")
else:
   print("The dataset is imbalanced.")

# Check distribution of predictions
predictions = classified_results['prediction'].value_counts()

print("Prediction distribution:")
print(predictions)

if predictions[0] > predictions[1]:
    print("Majority classified as human.")
else:
    print("Majority classified as AI.")



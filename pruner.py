import os
import pandas as pd

# Define the paths
csv_path = r"C:\Users\prapa\Documents\GitHub\AuscultationApp\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
base_folder = r"C:\Users\prapa\Documents\GitHub\AuscultationApp\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3"

# Read the CSV file to match with wav files
df = pd.read_csv(csv_path)

# List of categories and conditions
categories = {
    "No_murmur": (df["Murmur"] == "Absent"),
    "Systolic_murmur_abnormal": (df["Murmur"] == "Present") & (df["Systolic murmur grading"].notna()) & (df["Outcome"] == "Abnormal"),
    "Diastolic_murmur_abnormal": (df["Murmur"] == "Present") & (df["Diastolic murmur grading"].notna()) & (df["Outcome"] == "Abnormal"),
    "Systolic_murmur_normal": (df["Murmur"] == "Present") & (df["Systolic murmur grading"].notna()) & (df["Outcome"] == "Normal"),
    "Diastolic_murmur_normal": (df["Murmur"] == "Present") & (df["Diastolic murmur grading"].notna()) & (df["Outcome"] == "Normal"),
}

# Find all wav files
wav_files = []
for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".wav"):
            wav_files.append(os.path.join(root, file))
        else:
            # Delete non-wav files
            os.remove(os.path.join(root, file))

# Now we filter, prune, and rename the wav files
selected_files = []
for label, condition in categories.items():
    filtered_files = df[condition]["Patient ID"].dropna().astype(int).astype(str).tolist()
    category_files = [f for f in wav_files if any(fid in f for fid in filtered_files)]
    
    # Randomly select 20 files from each category
    selected_category_files = category_files[:20]
    
    # Rename files according to the label
    for file in selected_category_files:
        directory, original_filename = os.path.split(file)
        new_filename = f"{label}_{original_filename}"
        new_filepath = os.path.join(directory, new_filename)
        os.rename(file, new_filepath)
        selected_files.append(new_filepath)

# Prune by keeping only selected files and deleting the rest
for file in wav_files:
    if file not in selected_files:
        os.remove(file)

# Summary of actions
summary = {
    "total_wav_files": len(wav_files),
    "selected_files": len(selected_files)
}

print(summary)

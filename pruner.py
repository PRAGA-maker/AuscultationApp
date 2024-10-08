import os
import pandas as pd
import random

csv_path = r"C:\Users\prapa\Documents\GitHub\AuscultationApp\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
base_folder = r"C:\Users\prapa\Documents\GitHub\AuscultationApp\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3"

df = pd.read_csv(csv_path)

categories = {
    "No_murmur": (df["Murmur"] == "Absent"),
    "Systolic_murmur_abnormal": (df["Murmur"] == "Present") & (df["Systolic murmur grading"].notna()) & (df["Outcome"] == "Abnormal"),
    "Diastolic_murmur_abnormal": (df["Murmur"] == "Present") & (df["Diastolic murmur grading"].notna()) & (df["Outcome"] == "Abnormal"),
    "Systolic_murmur_normal": (df["Murmur"] == "Present") & (df["Systolic murmur grading"].notna()) & (df["Outcome"] == "Normal"),
    "Diastolic_murmur_normal": (df["Murmur"] == "Present") & (df["Diastolic murmur grading"].notna()) & (df["Outcome"] == "Normal"),
}

wav_files = []
for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".wav"):
            wav_files.append(os.path.join(root, file))
        else:
            os.remove(os.path.join(root, file))

selected_files = []
for label, condition in categories.items():
    filtered_files = df[condition]["Patient ID"].dropna().astype(int).astype(str).tolist()
    category_files = [f for f in wav_files if any(fid in f for fid in filtered_files)]
    selected_category_files = random.sample(category_files,10)
    for file in selected_category_files:
        directory, original_filename = os.path.split(file)
        new_filename = f"{label}_{original_filename}"
        new_filepath = os.path.join(directory, new_filename)
        os.rename(file, new_filepath)
        selected_files.append(new_filepath)

for file in wav_files:
    if file not in selected_files:
        os.remove(file)

summary = {
    "total_wav_files": len(wav_files),
    "selected_files": len(selected_files)
}

print(summary)

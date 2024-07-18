import os
import pandas as pd

def read_csv(folder_path):
    csv_dict = {}    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)            
            key = os.path.splitext(file_name)[0]  
            csv_dict[key] = pd.read_csv(file_path)
    return csv_dict

def calculate_sampling_rate(df):
    time_diffs = df['time'].diff().dropna()
    avg_time_diff = time_diffs.mean()
    sampling_rate = 1 / avg_time_diff
    return sampling_rate

import pandas as pd
import numpy as np

def determine_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

def determine_period(hour):
    if 6 <= hour < 10:
        return 'Morning'
    elif 11 <= hour < 14:
        return 'Noon'
    elif 14 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 23:
        return 'Evening'
    else:
        return 'Night'

def round_time_to_interval(entry_time, interval):
    minute_interval = interval // 60
    hour = entry_time.hour
    minute = entry_time.minute

    # Round down to the nearest interval
    rounded_minute = (minute // minute_interval) * minute_interval
    rounded_time = f"{hour:02d}:{rounded_minute:02d}:00"
    return rounded_time

# Load data
# file_path = 'E:/workspace/thesis data/mall/context_feature.txt'
file_path = 'E:/workspace/thesis data/mall/context_feature_latest5.txt'
data = pd.read_csv(file_path, sep='\t')

# Parse dates and times
#data['capture_date'] = pd.to_datetime(data['capture_date'], format='%d/%m/%Y')
# 修改1: 解析日期格式从 '%d/%m/%Y' 改为 '%Y-%m-%d'
data['capture_date'] = pd.to_datetime(data['capture_date'], format='%Y-%m-%d')
data['entry_time'] = pd.to_datetime(data['entry_time'], format='%H:%M:%S').dt.time

# Extract features
data['month'] = data['capture_date'].dt.month
data['weekday'] = data['capture_date'].dt.weekday
data['season'] = data['month'].apply(determine_season)
data['time_of_day'] = data['entry_time'].apply(lambda x: determine_period(x.hour))

# Time period (30 minutes interval as default, adjustable)
interval = 30 * 60  # 30 minutes in seconds
data['time_interval'] = data['entry_time'].apply(lambda x: round_time_to_interval(x, interval))

# Save the processed data
output_path = 'E:/workspace/thesis data/mall/context_feature_input.txt'
data.to_csv(output_path, sep='\t', index=False)
print('Processed data saved to', output_path)

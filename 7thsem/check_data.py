import pandas as pd
from datetime import datetime, timedelta

def parse_weird_date(date_str):
    # format: 30-03-2023 0.10
    try:
        date_part, time_part = date_str.split(' ')
        hours = int(float(time_part))
        minutes = int(round((float(time_part) - hours) * 100))
        # Handle cases like 1.58 meaning 1:58
        return datetime.strptime(f"{date_part} {hours:02d}:{minutes:02d}", "%d-%m-%Y %H:%M")
    except Exception as e:
        return None

df = pd.read_csv('../queue_data.csv')

# Parse Arrivals
df['parsed_arrival'] = df['arrival_time'].apply(parse_weird_date)

# Parse Finish (easy)
df['parsed_finish'] = pd.to_datetime(df['finish_time'])

# Calculate Service Duration
# Wait is in minutes? Let's check.
# Row 2: Arrival 00:10. Finish 00:22:44. Wait 12.68.
# 00:10 to 00:22:44 is 12 mins 44 sec = 12.73 mins.
# Wait 12.68 is almost 12.73.
# This implies Service Time was basically 0 (or negligible)?
# OR... Wait Time is Total System Time?
# "Wait Time" usually means time in queue. "System Time" is Wait + Service.
# If Wait ~ Total Time, then Service is small.

# Let's verify Row 30: 1.58 (01:58). Finish 02:21:26. 
# Total duration: 23 mins 26 sec = 23.4 mins.
# Wait: 22.82.
# Difference: 23.4 - 22.82 = 0.6 mins (36 seconds). 
# This implies service times are very short (~30-60 secs).

# Let's calculate inferred service time
df['total_system_time_min'] = (df['parsed_finish'] - df['parsed_arrival']).dt.total_seconds() / 60.0
df['inferred_service_min'] = df['total_system_time_min'] - df['wait_time']

print(df[['parsed_arrival', 'parsed_finish', 'wait_time', 'inferred_service_min']].head(10))
print("\nDescription of Service Times:")
print(df['inferred_service_min'].describe())

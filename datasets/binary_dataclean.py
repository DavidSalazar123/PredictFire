import numpy as np
import pandas as pd
import holidays

# Load Fire Deparment Document (Private File)
df = pd.read_csv("../original_dataset.csv")

# Call or not?
df.insert(1, "Call", 0)
df["Call"].where(df["IncidentTime"].isna(), 1, inplace=True)

# Insert One-Hot Encoding for time
time = pd.to_datetime(df["IncidentTime"]).dt.hour
df.insert(2, "Morning", np.where(
    (time > 5) & (time <= 12) & pd.notna(time), 1, 0))
df.insert(3, "Afternoon", np.where(
    (time > 12) & (time <= 17) & pd.notna(time), 1, 0))
df.insert(4, "Night", np.where((time > 17) & pd.notna(time), 1, 0))


# Dummies values for Conditions
one_hot_encode = pd.get_dummies(df["conditions"])
del df["conditions"]
df = df.join(one_hot_encode)

# Dummies values for dates
df["datetime"] = pd.to_datetime(df["datetime"])
one_hot_encode = pd.get_dummies(df["datetime"].dt.day_name())
df = df.join(one_hot_encode)

# Insert Holidays List
df.insert(2, "Holiday", 0)
usholidays = holidays.US()
df["Holiday"] = [1 if str(val).split()[0]
                 in usholidays else 0 for val in df["datetime"]]

# Format 
df = df.loc[:, list(df.columns[6:7])+list(df.columns[11:12])+list(df.columns[1:2])+list(df.columns[2:3])+list(df.columns[3:6])+list(df.columns[21:22])+list(df.columns[25:26])
       + list(df.columns[26:27])+list(df.columns[24:25])+list(df.columns[20:21])+list(df.columns[22:23])+list(df.columns[23:24])+list(df.columns[7:10])+list(df.columns[15:20])]

# Save cleaned data set
df.to_csv("./datasets/binarydataset.csv")

import numpy as np
import pandas as pd
import holidays

# Load Fire Deparment Document (Private File)
df = pd.read_csv("IncidentsData_Copy.csv")

# Matches dates and calls
df = pd.merge(
    df.iloc[:, 5:], df.iloc[:, :5], left_on="Date", right_on="IncidentDate", how="outer"
)
df.dropna(how="all", inplace=True)  # Remove Nan rows
del df["IncidentDate"]

# Call or not?
df.insert(1, "Call", 0)
df["Call"].where(df["Incident_Type"].isna(), 1, inplace=True)

# Fix Data for better wording
df["Conditions"].mask(
    df["Conditions"].astype(str).str.contains("Rain", case=False), "Rain", inplace=True
)  # Changes weather to rain if contains rain
df["Conditions"].mask(
    df["Conditions"].astype(str).str.contains("Cloudy", case=False),
    "Cloudy",
    inplace=True,
)  # Changes weather to rain if contains rain

# Insert One-Hot Encoding for time
time = pd.to_datetime(df["IncidentTime"]).dt.hour
df.insert(2, "Morning", np.where((time > 5) & (time <= 12) & pd.notna(time), 1, 0))
df.insert(3, "Afternoon", np.where((time > 12) & (time <= 17) & pd.notna(time), 1, 0))
df.insert(4, "Night", np.where((time > 17) & pd.notna(time), 1, 0))

# Dummies values for Conditions
one_hot_encode = pd.get_dummies(df["Conditions"])
del df["Conditions"]
df = df.join(one_hot_encode)

# Dummies values for dates
df["Date"] = pd.to_datetime(df["Date"])
one_hot_encode = pd.get_dummies(df["Date"].dt.day_name())
df = df.join(one_hot_encode)

# Insert Holidays List
df.insert(2, "Holiday", 0)
usholidays = holidays.US()
df["Holiday"] = [1 if str(val).split()[0] in usholidays else 0 for val in df["Date"]]
df = df.loc[
    :,
    list(df.columns[1:6])
    + list(df.columns[18:19])
    + list(df.columns[22:24])
    + list(df.columns[21:22])
    + list(df.columns[17:18])
    + list(df.columns[19:21])
    + list(df.columns[6:9])
    + list(df.columns[13:17]),
]

# Save cleaned data set
df.to_csv("./FirePredict_Dataset.csv")

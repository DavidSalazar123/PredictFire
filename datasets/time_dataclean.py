import numpy as np
import pandas as pd
import holidays

# Load Fire Deparment Document (Private File)
df = pd.read_csv("../original_dataset.csv")

# Call or not?
df.insert(1, "Call", 0)
df["Call"].where(df["IncidentTime"].isna(), 1, inplace=True)

# Grab actual fire calls
df = df[ df["Call"] == 1 ]

# Create a date time
df.insert(0,"DateTime", pd.to_datetime(df['Datetime'].astype(str) + " " + df['IncidentTime'].astype(str)))

# Created Time Series table for LSTM
df = df.loc[:,list(df.columns[0:1])+
list(df.columns[3:7])
+ list(df.columns[11:15])
]

# Save cleaned data set
df.to_csv("./datasets/LSTMdataset.csv")
# WEATHER DATA IS FROM : visualcrossing.com/
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

# Gather Data from calls
df = pd.read_csv("./Incidents.csv") # Private data
df = df[df['Address'].str.contains('Raritan', case=False) == True] #Only Raritan Calls
df = df.iloc[::-1] # Flip Dataframe
callsinfoDF = df[['IncidentDate','IncidentTime','IncidentType','Latitude','Longitude']]
callsinfoDF['IncidentDate'] = pd.to_datetime(df['IncidentDate'])

# Gather Data from weather table
df = pd.read_csv("./Weather.csv") # Private data
weatherDF = df[['datetime','temp','snowdepth','windspeed','conditions']]

# Fix Data for better wording
weatherDF["conditions"].mask(
    weatherDF["conditions"].astype(str).str.contains("Snow", case=False), "Snow", inplace=True
) 
weatherDF["conditions"].mask(
    weatherDF["conditions"].astype(str).str.contains("Rain", case=False), "Rain", inplace=True
)  # Changes weather to rain if contains rain
weatherDF["conditions"].mask(
    weatherDF["conditions"].astype(str).str.contains("Cloudy", case=False),
    "Cloudy",
    inplace=True,
) 
weatherDF['datetime'] = pd.to_datetime(weatherDF['datetime'])
weatherDF['snowdepth']=weatherDF['snowdepth'].fillna(0)

# Join weather and calls data
incidents_data = pd.merge(weatherDF, callsinfoDF, left_on='datetime',right_on='IncidentDate',how='outer')
incidents_data.iloc[::-1]

# Save cleaned data set
incidents_data.to_csv("./datasets/original_dataset.csv")
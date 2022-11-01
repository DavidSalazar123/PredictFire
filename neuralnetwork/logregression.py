import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
def main():

    #Stacked CSV
    csvfile_stacked = pd.read_csv("IncidentInfo_Stacked.csv")
    dataset_stacked = csvfile_stacked[["Call", "Time_of_Day", "Season", "Month_Number", "Weekday",
                                        "Holiday", "Weather", "Temp", "Snow_Depth", "Wind_Speed"]]
    headers_stacked = csvfile_stacked.loc[:, ~csvfile_stacked.columns.isin(['Date', "Call"])]
    function_stacked = " + ".join(headers_stacked) #String/function formating

    #Unstacked CSV
    csvfile_unstacked = pd.read_csv("IncidentInfo_Spread.csv", index_col = 0)
    headers_unstacked = np.array(csvfile_unstacked.columns)
    headers_unstacked_function = csvfile_unstacked.loc[:, csvfile_unstacked.columns!='Call']
    dataset_unstacked = csvfile_unstacked[headers_unstacked]
    function_unstacked = " + ".join(headers_unstacked_function) #String/function formating

    #dependent variables ~ (Given) independent variables
    log_reg_stacked = sm.mnlogit("Call ~ " + function_stacked, data = dataset_stacked,).fit()
    log_reg_unstacked = sm.mnlogit("Call ~ " + function_unstacked, data=dataset_unstacked).fit()

    coefficients_stacked = log_reg_stacked.params.values.tolist() #Converts coefficients to list for inputs
    coefficients_unstacked = log_reg_unstacked.params.values.tolist() #Converts coefficients to list for inputs

    print(log_reg_unstacked.summary())


if __name__ == '__main__':
    main()
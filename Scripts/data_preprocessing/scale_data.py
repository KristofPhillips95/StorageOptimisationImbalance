import pandas as pd
import h5py


def retrieve_name(key, col, dict_col_rename):
    if key == "Global_data_Forecast_Historical_CIPU":
        return col+"_fc"
    elif key == "Global_data_generation_produced":
        return col+"_act"
    else:
        if col in dict_col_rename:
            return dict_col_rename[col]
        else:
            return col



df_scaling = pd.read_excel("Scaling_values.xlsx")


loc = "h5_files/20240910_SI.h5"
f = h5py.File(loc)
ks = f.keys()


dict_col_scaling_values = {
    "ACE": "ACE",
    "SI": "ACE",
    'Imb_price': 'Imb_price',
    'alpha': 'Imb_price',
    "PV_fc": "PV",
    "PV_act": "PV",
    "wind_fc": "W_total",
    "wind_act": "W_total",
    "Gas_fc": "Gas",
    "Gas_act": "Gas",
    "Nuclear_fc": "Nuclear",
    "Nuclear_act": "Nuclear",
    "Water_fc": "Water",
    "Water_act": "Water",
    "load_act": "Load",
    "load_fc": "Load",
    "BEDE": "Net_pos",
    "BEFR": "Net_pos",
    "BELU": "Net_pos",
    "BENL": "Net_pos",
    "BEUK": "Net_pos",
    "Net_Position": "Net_pos",
    "-Max": "MO_d",
    "Max": "MO_u"
}

#Add merit order

for i in range(10):
    dict_col_scaling_values[f"-{int(100*(i+1))}MW"] = "MO_d"
    dict_col_scaling_values[f"{int(100*(i+1))}MW"] = "MO_u"




dict_col_rename = {
    "Day-ahead Price [€\MWh]": "DAP",
    "ACE": "ACE",
    "SI": "SI",
    "DA_pv_MW": "PV_fc",
    "RT_pv_MW": "PV_act",
    "DA_wind_total_MW": "wind_fc",
    "RT_wind_total_MW": "wind_act",
    "Total_Load_real_values": "load_act",
    "Total_Load_forecasted_values": "load_fc",
    "alpha [€/MWh]": "alpha"
}


dict_cols = {
    #"DATA_DA_Prices": ["Day-ahead Price [€\MWh]"], #TODO: adjust DAP data to quarter hourly basis
    "DATA_ELIA_imbalancenrvprices":["ACE","SI","Imb_price","alpha [€/MWh]"],
    "DATA_ELIA_pv":["DA_pv_MW", "RT_pv_MW"],
    "DATA_ELIA_wind_total":["DA_wind_total_MW", "RT_wind_total_MW"],
    "Global_data_Forecast_Historical_CIPU":["Gas","Nuclear","Water"],
    "Global_data_generation_produced":["Gas","Nuclear","Water"],
    "Total_Load":["Total_Load_real_values", "Total_Load_forecasted_values"],
    "data_ELIA_flow":["BEDE","BEFR","BELU","BENL","BEUK","Net_Position"],
    #"DATA_ELIA_MO": [f"-{int(100*(i+1))}MW" for i in range(10)] + [f"{int(100*(i+1))}MW" for i in range(10)]
}




list_dfs = []
for key,cols in dict_cols.items():

    df = pd.read_hdf(loc,key).loc[:,cols]
    list_dfs.append(df)

    for col in cols:
        print(col)
        if col == '-100MW':
            x=1
        col_name = retrieve_name(key, col, dict_col_rename)
        df.rename(columns={col:col_name},inplace=True)
        scale_row = dict_col_scaling_values[col_name]
        min = df_scaling.loc[df_scaling["Unnamed: 0"] == scale_row, "Min"].values[0]
        max = df_scaling.loc[df_scaling["Unnamed: 0"] == scale_row, "Max"].values[0]
        df[col_name] = (df[col_name] - min)/(max-min)

    try:
        df_large = df_large.join(df)
    except:
        df_large=df



df_large_filtered = df_large[df_large.index < pd.to_datetime('2024-01-01')]
rows_with_missing_values = df_large_filtered[df_large_filtered.isnull().any(axis=1)]



df_large.to_hdf("data_qh_SI_imbPrice_scaled_alpha.h5",key="data",mode="w")




x=1
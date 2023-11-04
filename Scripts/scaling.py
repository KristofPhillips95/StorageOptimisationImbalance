import pandas as pd
import h5py

def scale_data(df=None,loc=None,ks=None):

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

    dict_col_scaling_values = {
        "ACE": "ACE",
        "SI": "ACE",
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
        "Net_Position": "Net_pos"

    }

    dict_col_rename = {
        "Day-ahead Price [€\MWh]": "DAP",
        "ACE": "ACE",
        "SI": "SI",
        "DA_pv_MW": "PV_fc",
        "RT_pv_MW": "PV_act",
        "DA_wind_total_MW": "wind_fc",
        "RT_wind_total_MW": "wind_act",
        "Total_Load_real_values": "load_act",
        "Total_Load_forecasted_values": "load_fc"
    }


    dict_cols = {
        #"DATA_DA_Prices": ["Day-ahead Price [€\MWh]"], #TODO: adjust DAP data to quarter hourly basis
        "DATA_ELIA_imbalancenrvprices":["ACE","SI"],
        "DATA_ELIA_pv":["DA_pv_MW", "RT_pv_MW"],
        "DATA_ELIA_wind_total":["DA_wind_total_MW", "RT_wind_total_MW"],
        "Global_data_Forecast_Historical_CIPU":["Gas","Nuclear","Water"],
        "Global_data_generation_produced":["Gas","Nuclear","Water"],
        "Total_Load":["Total_Load_real_values", "Total_Load_forecasted_values"],
        "data_ELIA_flow":["BEDE","BEFR","BELU","BENL","BEUK","Net_Position"]
    }

    def scale_col(df,col,rename=True):
        if rename:
            col_name = retrieve_name(key, col, dict_col_rename)
            df.rename(columns={col: col_name}, inplace=True)
        else:
            col_name=col
        scale_row = dict_col_scaling_values[col_name]
        min = df_scaling.loc[df_scaling["Unnamed: 0"] == scale_row, "Min"].values[0]
        max = df_scaling.loc[df_scaling["Unnamed: 0"] == scale_row, "Max"].values[0]
        df[col_name] = (df[col_name] - min) / (max - min)

        return df


    if df is None:
        list_dfs = []
        for key,cols in dict_cols.items():

            df = pd.read_hdf(loc,key).loc[:,cols]
            list_dfs.append(df)

            for col in cols:
                df = scale_col(df,col)

            try:
                df_large = df_large.join(df)
            except:
                df_large=df

    else:
        cols = df.columns

        for col in cols:
            df=scale_col(df,col,rename=False)

        df_large=df



    return df_large






if __name__ == "__main__":
    loc = "DATA_SI_15T.h5"
    f = h5py.File(loc)
    ks = f.keys()

    df_large = scale_df(loc=loc,ks=ks)

    df_large.to_hdf("data_scaled.h5",key="data",mode="w")

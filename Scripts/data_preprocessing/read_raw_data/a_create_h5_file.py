import pandas as pd
from datetime import datetime


import quarter_hourly_data_MO as read_mo
import quarter_hourly_data_load as read_load
import quarter_hourly_data_res as read_res
import quarter_hourly_data_imbalancenrvprices as read_imbalance
import quarter_hourly_data_generation_forecast as read_gen
import quarter_hourly_data_flow as read_flow




#Initialization
name = '20240910_SI'
start_date = datetime(2014, 1, 1)
end_date = datetime(2024,1,1)

dp_to_include = [
    'flow',
    #'gen_fc',
    #'gen_act',
    #'imb_nrv_price',
    #'pv',
    #'wind_on',
    #'wind_off',
    #'wind_tot',
    #'load',
    #'ARC_MO'
]

cp = pd.DataFrame({
    'start_trainingData': [start_date],
    'end_trainingData': [end_date],
    'file_h5': [f'../h5_files/{name}.h5'],
    'loc_data': ['C:/Users/u0137781/OneDrive - KU Leuven/data/raw']
})

with pd.HDFStore(f'../h5_files/{name}.h5', mode='a') as store:  # 'a' mode allows appending/modifying without overwriting the file
    del store['data_ELIA_flow']

if 'flow' in dp_to_include:
    read_flow.import_quarter_hourly_data_flow(cp)
if 'gen_fc' in dp_to_include:
    read_gen.import_quarter_hourly_data_forecast_generation(cp)
if 'gen_act' in dp_to_include:
    read_gen.import_quarter_hourly_data_produced_generation(cp)
if 'imb_nrv_price' in dp_to_include:
    read_imbalance.import_quarter_hourly_data_imbalancenrvprices(cp)
if 'pv' in dp_to_include:
    read_res.import_quarter_hourly_data_pv(cp)
if 'wind_on' in dp_to_include:
    read_res.import_quarter_hourly_data_wind_on_shore(cp)
if 'wind_off' in dp_to_include:
    read_res.import_quarter_hourly_data_wind_off_shore(cp)
if 'wind_tot' in dp_to_include:
    read_res.import_quarter_hourly_data_wind_total(cp)
if 'load' in dp_to_include:
    read_load.import_quarter_hourly_data_load(cp)
if 'ARC_MO' in dp_to_include:
    read_mo.import_quarter_hourly_MO(cp)
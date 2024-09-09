import pandas as pd
from datetime import datetime


import quarter_hourly_data_MO as read_mo
import quarter_hourly_data_load as read_load
import quarter_hourly_data_res as read_res
import quarter_hourly_data_imbalancenrvprices as read_imbalance
import quarter_hourly_data_generation_forecast as read_gen
import quarter_hourly_data_flow as read_flow


#Initialization
name = '20240909_SI'
start_date = datetime(2022, 1, 1)
end_date = datetime(2024,1,1)

cp = pd.DataFrame({
    'start_trainingData': [start_date],
    'end_trainingData': [end_date],
    'file_h5': [f'../h5_files/{name}.h5'],
    'loc_data': ['C:/Users/u0137781/OneDrive - KU Leuven/data/raw']
})

#read_gen.import_quarter_hourly_data_forecast_generation(cp)
read_flow.import_quarter_hourly_data_flow(cp)

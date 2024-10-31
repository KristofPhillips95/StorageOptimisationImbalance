import pandas as pd
from datetime import date, time, datetime, timedelta
from dateutil.rrule import rrule, MONTHLY, YEARLY
import os
import math
# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
# Change directory
os.chdir(cwd)


def import_quarter_hourly_data_flow(config_parameters):
    slash_name_border = ['\BEFR','\BEDE','\BELU','\BENL','\BEUK']
    name_border = ['BEFR','BEDE','BELU','BENL','BEUK']
    file_h5 = config_parameters['file_h5'][0]
    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    #Time change in spring
    #Time change in fall
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H
    index = pd.date_range(start_trainingData, end_trainingData, freq="15min")
    df_all = pd.DataFrame(0,columns=name_border, index=index, dtype=float)
    print(df_all)

    for j in range(len(name_border)):
        file_xlsx =f"{config_parameters['loc_data'][0]}/Elia/Flows/{name_border[j]}/Global_data.xlsx"
        dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]
        df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE','PhysicalFlowValue'], dtype=float)
        for i in range(len(dates_loop)):
            print(dates_loop[i])
            df_1 = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
            n_year = dates_loop[i].year
            n_month = '{:02d}'.format(dates_loop[i].month)
            file = f"{config_parameters['loc_data'][0]}/Elia/Flows/{name_border[j]}/PhysicalFlow_{name_border[j]}_{str(n_year)}{str(n_month)}.xlsx"
            if os.path.isfile(file) == True:
                # Load excel file
                print(file)
                file_unprocessed = pd.read_excel(file, header=0, skiprows=4, engine = 'openpyxl')
                if 'Physical flow on the Belgium-France border [MW]' in file_unprocessed.columns:
                    file_unprocessed.rename(columns={'Physical flow on the Belgium-France border [MW]': 'PhysicalFlowValue'}, inplace=True)
                if 'Physical flow on the Belgium-Luxembourg border [MW]' in file_unprocessed.columns:
                    file_unprocessed.rename(columns={'Physical flow on the Belgium-Luxembourg border [MW]': 'PhysicalFlowValue'}, inplace=True)
                if 'Physical flow on the Belgium-Netherlands border [MW]' in file_unprocessed.columns:
                    file_unprocessed.rename(columns={'Physical flow on the Belgium-Netherlands border [MW]': 'PhysicalFlowValue'}, inplace=True)
                if 'Physical flow on the Belgium-United Kingdom border [MW]' in file_unprocessed.columns:
                    file_unprocessed.rename(columns={'Physical flow on the Belgium-United Kingdom border [MW]': 'PhysicalFlowValue'}, inplace=True)

                if 'Datetime (CET+1/CEST +2)' in file_unprocessed.columns:
                    file_unprocessed.rename(columns={'Datetime (CET+1/CEST +2)': 'Date/Time'}, inplace=True)
                    #The following try/except construction exists because there is an issue with the datetime format in the file "PhysicalFlow_BELU_202207.xlsx"
                    try:
                        file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['Date/Time'], format='%d/%m/%Y %H:%M')
                    except:
                        file_unprocessed['Date/Time'] = pd.to_datetime(file_unprocessed['Date/Time'], format='%d/%m/%Y %H:%M:%S')
                        file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['Date/Time'], format='%d/%m/%Y %H:%M')



                else:

                    try:
                        file_unprocessed['Date/Time'] = file_unprocessed['Date/Time'].str.replace('/', '-')
                    except:
                        print('except')

                    file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['Date/Time'], format='%d-%m-%Y %H:%M:%S')#, format='%d/%m/%Y %H:%M'
                file_unprocessed.set_index('DateTime',inplace=True) # column Date/Time as index
                file_unprocessed = file_unprocessed[file_unprocessed.index.month == dates_loop[i].month]  # select only the beginning of each day

                print(file_unprocessed)
                percent_missing = file_unprocessed.isnull().sum() * 100 / len(file_unprocessed) # percentage of missing data per column
                print( 'Missing data (%) : PhysicalFlowValue '+str(round(percent_missing['PhysicalFlowValue'],2)) )
                if round(percent_missing['PhysicalFlowValue'],2) == 100:
                    print(file_unprocessed)
                try:
                    file_unprocessed = file_unprocessed.resample('15min').mean().interpolate(method='linear',limit_direction='both') # Deals with spring hour and fall hour.
                except:
                    x=1
                    y=2
                for date_dst_fall in DST_in_fall:
                    if date_dst_fall in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ') # resample seems to do the job for removing the added hour in fall
                        #print(file_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                for date_dst_spring in DST_in_spring:
                    if date_dst_spring in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ') # resample seems to do the job for interpolating the missing hour in spring
                        #print(file_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification

                df_1 = df_1.assign(FROM_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(TO_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(PhysicalFlowValue=file_unprocessed['PhysicalFlowValue'].values)
                df = pd.concat([df, df_1], ignore_index=True)
            else:
                print('This file doesn t exist',file)
        df = df.set_index('FROM_DATE')
        df_all.loc[df.index,name_border[j]] = df['PhysicalFlowValue'].values
    df_all['Net_Position'] = df_all.sum(axis=1)
    print(df_all.shape)
    print(df_all.head())
    print(df_all.tail())
    # Write CSV
    open(file_xlsx, 'w+')
    df_all.to_excel(file_xlsx)
    # break
    df_all.to_hdf(file_h5, 'data_ELIA_flow')





if __name__ == '__main__':
    config_parameters = pd.DataFrame({'start_trainingData': [datetime(2014, 1, 1)],
                                      'end_trainingData': [datetime(2024, 1, 1)],
                                      'file_h5': 'data_qh_SI_imbPrice.h5',
                                      })
    import_quarter_hourly_data_flow(config_parameters)


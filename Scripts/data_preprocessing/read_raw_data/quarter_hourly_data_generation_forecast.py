import pandas as pd
from datetime import date, time, datetime, timedelta
from dateutil.rrule import rrule, MONTHLY, YEARLY
import os
from pandas.api.types import is_object_dtype
import numpy as np
# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
# Change directory
os.chdir(cwd)
print(cwd)

def import_quarter_hourly_data_forecast_generation(config_parameters):
    #file_xlsx = r'''file csv/DATA Elia/Generation/Global_data_Forecast_Historical_CIPU.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/Generation/Global_data_Forecast_Historical_CIPU.xlsx"
    file_h5 = config_parameters['file_h5'][0]

    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H

    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]

    #df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE','Total_without_coal','Fuel','Gas','Nuclear','Water','Wind','Other' ], dtype=float)#], dtype=float)
    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE','Gas','Nuclear','Water' ], dtype=float)#], dtype=float)
    #sheet = ['Sheet2','Sheet3','Sheet4','Sheet5','Sheet6','Sheet7']
    sheet = ['Sheet3','Sheet4','Sheet5']
    production= ['Gas','Water','Nuclear']


    for i in range(len(dates_loop)):
        print(dates_loop[i])
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file = f"{config_parameters['loc_data'][0]}/Elia/Generation/Generation_Forecast_Historical_Cipu_{n_month}_{n_year}.XLS"
        #file = r'''file csv\DATA Elia\Generation\Generation_Forecast_Historical_Cipu_'''+str(n_month)+'''_'''+str(n_year)+'''.XLS'''
        #file_unprocessed_all_sheet = pd.DataFrame(columns=['Total_without_coal',  'Fuel', 'Gas', 'Nuclear', 'Water', 'Wind', 'Other', 'FROM_DATE', 'TO_DATE'])  # ], dtype=float)
        file_unprocessed_all_sheet = pd.DataFrame(columns=['Gas', 'Nuclear', 'Water','FROM_DATE', 'TO_DATE'])  # ], dtype=float)
        if os.path.isfile(file) == True:
            for name_sheet,name_production in zip(sheet,production):
                # Load excel file
                file_unprocessed_sheet = pd.read_excel(file, header=0, skiprows=1,sheet_name=name_sheet)
                file_unprocessed_sheet = file_unprocessed_sheet.drop(file_unprocessed_sheet.index[0])  # Removing first line
                file_unprocessed_sheet['Date']=file_unprocessed_sheet['dd'].astype(int).astype(str)+'/'+file_unprocessed_sheet['Unnamed: 2'].astype(int).astype(str)\
                                                 +'/'+file_unprocessed_sheet['Unnamed: 5'].astype(int).astype(str)
                file_unprocessed_sheet.set_index([file_unprocessed_sheet['Date']],inplace=True)  # Index of date (%Y/%M/%D)
                file_unprocessed_sheet.drop(['mm', 'Unnamed: 3', 'yyyy', 'Unnamed: 6', 'Unnamed: 5', 'Unnamed: 2', 'dd','Date',], axis=1, inplace=True)
                if "*2:00 -> *2:15" in file_unprocessed_sheet.columns:
                    file_unprocessed_sheet.drop(["*2:00 -> *2:15", "*2:15 -> *2:30", "*2:30 -> *2:45",'*2:45 -> *3:00'], axis=1,inplace=True)  # Removing the added hour in fall
                file_unprocessed_sheet = file_unprocessed_sheet.stack(future_stack=True)  # Multiple index of date (%Y/%M/%D) and hour (%H/%M) as index
                file_unprocessed_sheet = file_unprocessed_sheet.reset_index(level=[0, 1])
                file_unprocessed_sheet["period"] = file_unprocessed_sheet["Date"]+ " " + file_unprocessed_sheet["level_1"].str.split(' ').str[0]  # create a timestamp in a correct format
                file_unprocessed_sheet.drop(['Date', 'level_1'], axis=1, inplace=True)  # remove useless columns
                file_unprocessed_sheet['period'] = pd.to_datetime(file_unprocessed_sheet['period'],format='%d/%m/%Y %H:%M')  # column period in datetime format
                file_unprocessed_sheet.set_index('period', inplace=True)  # index of date (%Y/%M/%D %H:%M)
                file_unprocessed_sheet.rename(columns={0: name_production},inplace=True)  # Renaming the column to Total_Load
                percent_missing = file_unprocessed_sheet.isnull().sum() * 100 / len(file_unprocessed_sheet)  # percentage of missing data per column
                print('Missing data (%) : '+name_production+' '+ str(round(percent_missing[name_production], 2)))
                file_unprocessed_sheet = file_unprocessed_sheet.resample('15min').mean().interpolate(limit_direction='both')  # Deals with spring hour and fall hour.
                for date_dst_fall in DST_in_fall:
                    if date_dst_fall in file_unprocessed_sheet.index.get_level_values('period'):
                        print('OK ')  # resample seems to do the job for removing the added hour in fall
                        #print(file_unprocessed_sheet.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                for date_dst_spring in DST_in_spring:
                    if date_dst_spring in file_unprocessed_sheet.index.get_level_values('period'):
                        print('OK ')  # resample seems to do the job for interpolating the missing hour in spring
                        #print(file_unprocessed_sheet.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                file_unprocessed_all_sheet[name_production] = file_unprocessed_sheet[name_production]
            #file_unprocessed_all_sheet[production_missing]=file_unprocessed_all_sheet[production[0]]+file_unprocessed_all_sheet[production[1]]+\
                #                                               file_unprocessed_all_sheet[production[2]]+ file_unprocessed_all_sheet[production[3]]+ \
            #                                           file_unprocessed_all_sheet[production[4]] + file_unprocessed_all_sheet[production[5]]
            file_unprocessed_all_sheet['FROM_DATE'] = [dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed_sheet.shape[0])]
            file_unprocessed_all_sheet["TO_DATE"]=[dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed_sheet.shape[0])]
            df = df._append(file_unprocessed_all_sheet, ignore_index=True, sort=True)
            print(df.shape)
        else:
            print('This file doesn t exist',file)
    df = df.set_index('FROM_DATE')
    #print(df.shape)
    #print(df.head())
    #print(df.tail())
    # Ecritrue fichier CSV
    open(file_xlsx, 'w+')
    df.to_excel(file_xlsx)
    # break
    df.to_hdf(file_h5, 'Global_data_Forecast_Historical_CIPU')


def import_quarter_hourly_data_produced_generation(config_parameters):
    #file_xlsx =r'''file csv/DATA Elia/Generation/Global_data_generation_produced.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/Generation/Global_data_Forecast_Historical_CIPU.xlsx"

    file_h5 = config_parameters['file_h5'][0]

    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0)] # + 1 H

    DST_fall_2021 = datetime(2021,10,31,0,0,0)
    DST_spring_2021 = datetime(2021,3,28,0,0,0)

    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]

    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE' ], dtype=float)#], dtype=float)
    sheet = ['Sheet4','Sheet5','Sheet6']
    production= ['Gas','Nuclear','Water']


    for i in range(len(dates_loop)):
      #if i == 9:
        print(dates_loop[i])
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file = f"{config_parameters['loc_data'][0]}/Elia/Generation/Generation_Produced_Historical_{n_year}-{n_month}.XLS"
        #file = r'''file csv\DATA Elia\Generation\Generation_Produced_Historical_'''+str(n_year)+'''-'''+str(n_month)+'''.XLS'''
        file_unprocessed_all_sheet = pd.DataFrame(columns=['Gas', 'Nuclear','Water','FROM_DATE', 'TO_DATE']) # ], dtype=float)

        if n_year < 2021:
            if os.path.isfile(file) == True:
                for name_sheet,name_production in zip(sheet,production):
                    # Load excel file
                    file_unprocessed_sheet = pd.read_excel(file, header=0, skiprows=3,sheet_name=name_sheet)
                    file_unprocessed_sheet = file_unprocessed_sheet.drop(file_unprocessed_sheet.index[0])  # Removing first line
                    file_unprocessed_sheet['Date']=file_unprocessed_sheet['dd'].astype(int).astype(str)+'/'+file_unprocessed_sheet['Unnamed: 2'].astype(int).astype(str)\
                                                     +'/'+file_unprocessed_sheet['Unnamed: 4'].astype(int).astype(str)
                    #print(file_unprocessed_sheet['Date'].head(5))
                    file_unprocessed_sheet.set_index([file_unprocessed_sheet['Date']],inplace=True)  # Index of date (%Y/%M/%D)
                    file_unprocessed_sheet.drop(['mm',  'yyyy',   'Unnamed: 2','Unnamed: 4','Unnamed: 5', 'dd','Date',], axis=1, inplace=True)

                    if "02:00 bis" in file_unprocessed_sheet.columns:
                        print('I am here 2')
                        file_unprocessed_sheet.drop(["02:00 bis", "02:15 bis", "02:30 bis",'02:45 bis'], axis=1,inplace=True)  # Removing the added hour in fall
                    file_unprocessed_sheet = file_unprocessed_sheet.stack(dropna=False)  # Multiple index of date (%Y/%M/%D) and hour (%H/%M) as index
                    file_unprocessed_sheet = file_unprocessed_sheet.reset_index(level=[0, 1])
                    file_unprocessed_sheet["period"] = file_unprocessed_sheet["Date"]+ " " + file_unprocessed_sheet["level_1"].str.split(' ').str[0]  # create a timestamp in a correct format
                    file_unprocessed_sheet.drop(['Date', 'level_1'], axis=1, inplace=True)  # remove useless columns
                    file_unprocessed_sheet['period'] = pd.to_datetime(file_unprocessed_sheet['period'],format='%d/%m/%Y %H:%M')  # column period in datetime format
                    file_unprocessed_sheet['period'] = file_unprocessed_sheet['period'] - pd.Timedelta(hours=0.25)  # column period in datetime format
                    file_unprocessed_sheet.set_index('period', inplace=True)  # index of date (%Y/%M/%D %H:%M)
                    #print(file_unprocessed_sheet.head(5))
                    file_unprocessed_sheet.rename(columns={0: name_production},inplace=True)  # Renaming the column to Total_Load
                    if is_object_dtype(file_unprocessed_sheet[name_production]):
                        #print(file_unprocessed_sheet[name_production].str.contains("NOT VALID", na=False).any())
                        indice_spring= file_unprocessed_sheet[file_unprocessed_sheet[name_production].str.contains("NOT VALID",na=False)].index.values
                        file_unprocessed_sheet[name_production].loc[indice_spring] = [np.nan for _ in range(len(indice_spring))]
                    file_unprocessed_sheet = file_unprocessed_sheet.astype(float)
                    percent_missing = file_unprocessed_sheet.isnull().sum() * 100 / len(file_unprocessed_sheet)  # percentage of missing data per column
                    print('Missing data (%) : '+name_production+' '+ str(round(percent_missing[name_production], 2)))
                    file_unprocessed_sheet = file_unprocessed_sheet.interpolate()  # Deals with spring hour and fall hour.
                    for date_dst_fall in DST_in_fall:
                        if date_dst_fall in file_unprocessed_sheet.index.get_level_values('period'):
                            print('OK ')  # resample seems to do the job for removing the added hour in fall
                            #print(file_unprocessed_sheet.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                    for date_dst_spring in DST_in_spring:
                        if date_dst_spring in file_unprocessed_sheet.index.get_level_values('period'):
                            print('OK ')  # resample seems to do the job for interpolating the missing hour in spring
                            #print(file_unprocessed_sheet.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                    file_unprocessed_all_sheet[name_production] = file_unprocessed_sheet[name_production]
                file_unprocessed_all_sheet['FROM_DATE'] = [dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed_sheet.shape[0])]
                file_unprocessed_all_sheet["TO_DATE"]=[dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed_sheet.shape[0])]

                df = df._append(file_unprocessed_all_sheet, ignore_index=True,sort=True)
                print(df.shape)
            else:
                print('This file doesn t exist',file)
        else: #if year >= 2021
            file_unprocessed = pd.read_excel(file, header=0, skiprows=4)
            file_unprocessed_all_sheet['period'] = file_unprocessed['Datetime (CET+1/CEST +2)']
            file_unprocessed_all_sheet['Gas'] = file_unprocessed['Natural Gas [MW]']
            file_unprocessed_all_sheet['Nuclear'] = file_unprocessed['Nuclear [MW]']
            file_unprocessed_all_sheet['Water'] = file_unprocessed['Water [MW]']

            if n_month == str(10):
                day_of_change = DST_fall_2021.day
                start_index_drop = 96*(day_of_change-1)+8
                file_unprocessed_all_sheet = file_unprocessed_all_sheet.drop(file_unprocessed_all_sheet.index[range(start_index_drop,start_index_drop+4)])

            if n_month == '0'+str(3):
                day_of_change = DST_spring_2021.day
                start_index_insert = 96*(day_of_change-1)+8
                avg_water = (file_unprocessed_all_sheet.loc[start_index_insert-1]['Water'] + file_unprocessed_all_sheet.loc[start_index_insert+1]['Water']) /2
                avg_gas = (file_unprocessed_all_sheet.loc[start_index_insert - 1]['Gas'] + file_unprocessed_all_sheet.loc[start_index_insert]['Gas']) / 2
                avg_nuclear = (file_unprocessed_all_sheet.loc[start_index_insert - 1]['Nuclear'] + file_unprocessed_all_sheet.loc[start_index_insert]['Nuclear']) / 2

                for qh in range(4):
                    dt_row = DST_spring_2021 + timedelta(days = 0, hours = 2, minutes = 15*qh)
                    row_values = [avg_gas,avg_nuclear,avg_water,0,0,dt_row]
                    row_series = pd.Series(row_values,index=file_unprocessed_all_sheet.columns)
                    file_unprocessed_all_sheet = file_unprocessed_all_sheet._append(row_series,ignore_index=True)

            file_unprocessed_all_sheet.sort_values(by='period', inplace=True)
            file_unprocessed_all_sheet.set_index('period', inplace=True)

            file_unprocessed_all_sheet['FROM_DATE'] = [dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed_all_sheet.shape[0])]
            file_unprocessed_all_sheet["TO_DATE"] = [dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed_all_sheet.shape[0])]

            df = df._append(file_unprocessed_all_sheet, ignore_index=True, sort=True)
            print(df.shape)


    df = df.set_index('FROM_DATE')
    #print(df.shape)
    #print(df.head())
    #print(df.tail())
    # Ecritrue fichier CSV
    open(file_xlsx, 'w+')
    df.to_excel(file_xlsx)
    # break
    df.to_hdf(file_h5, 'Global_data_generation_produced')





if __name__ == '__main__':
    config_parameters = pd.DataFrame({'start_trainingData': [datetime(2014, 1, 1)],
                                      'end_trainingData': [datetime(2024, 2, 29)],
                                      'file_h5': 'data_qh_SI_imbPrice.h5',
                                      })
    import_quarter_hourly_data_forecast_generation(config_parameters)
    import_quarter_hourly_data_produced_generation(config_parameters)

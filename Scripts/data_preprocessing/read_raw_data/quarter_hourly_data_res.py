import pandas as pd
from datetime import date, time, datetime, timedelta
from dateutil.rrule import rrule, MONTHLY, YEARLY
import os
import math
# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
# Change directory
os.chdir(cwd)

#################################
#PV
####################################
def import_quarter_hourly_data_pv(config_parameters):
    file_xlsx =r'''file csv\DATA Elia\RES\pv\Global_data.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/RES/pv/Global_data.xlsx"
    file_h5 = config_parameters['file_h5'][0]


    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE', 'DA_pv_MW','RT_pv_MW', 'Capacity_pv_MWp'], dtype=float)
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H

    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]
    for i in range(len(dates_loop)):
        print(dates_loop[i])
        df_1 = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file = r'''file csv\DATA Elia\RES\pv\SolarForecast_'''+str(n_year)+'''-'''+str(n_month)+'''.xls'''
        file = f"{config_parameters['loc_data'][0]}/Elia/RES/pv/SolarForecast_{n_year}-{n_month}.xls"

        if os.path.isfile(file) == True:
            # Load excel file
            file_unprocessed = pd.read_excel(file, header=0, skiprows=3)
            file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['DateTime'], format='%d/%m/%Y %H:%M')
            file_unprocessed.set_index('DateTime',inplace=True) # column Date/Time as index
            file_unprocessed = file_unprocessed[file_unprocessed.index.month == dates_loop[i].month]  # select only the beginning of each day
            if 'Week-Ahead forecast [MW]' in file_unprocessed.columns:
                file_unprocessed.drop(['Week-Ahead forecast [MW]'], axis=1, inplace=True)
            if 'Most recent forecast [MW]' in file_unprocessed.columns:
                file_unprocessed.rename(columns={'Most recent forecast [MW]': 'Intraday forecast [MW]'}, inplace=True)
            percent_missing = file_unprocessed.isnull().sum() * 100 / len(file_unprocessed) # percentage of missing data per column
            print( 'Missing data (%) : DA '+str(round(percent_missing['Day-Ahead forecast [MW]'],2))+', RT '+str(round(percent_missing['Real-time Upscaled Measurement [MW]'],2)) )
            file_unprocessed = file_unprocessed.resample('15min').mean().interpolate(limit_direction='both') # Deals with spring hour and fall hour.
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
            df_1 = df_1.assign(DA_pv_MW=file_unprocessed['Day-Ahead forecast [MW]'].values)
            df_1 = df_1.assign(RT_pv_MW=file_unprocessed['Real-time Upscaled Measurement [MW]'].values)
            df_1 = df_1.assign(Capacity_pv_MWp=file_unprocessed['Monitored Capacity [MWp]'].values)
            df = df._append(df_1, ignore_index=True)
        else:
            print('This file doesn t exist',file)
    df = df.set_index('FROM_DATE')
    print(df.shape)
    print(df.head())
    print(df.tail())
    # Ecritrue fichier CSV
    open(file_xlsx, 'w+')
    df.to_excel(file_xlsx)
    # break
    df.to_hdf(file_h5, 'DATA_ELIA_pv')


##############################################################################
#Wind OFF shore
########################################################################
def import_quarter_hourly_data_wind_off_shore(config_parameters):
    file_xlsx = r'''file csv\DATA Elia\RES\Wind off shore\Global_data.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/RES/Wind off shore/Global_data.xlsx"

    file_h5 = config_parameters['file_h5'][0]

    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE', 'DA_wind_off_shore_MW', 'RT_wind_off_shore_MW','Capacity_wind_off_shore_MW'], dtype=float)
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H

    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]

    for i in range(len(dates_loop)):
        print(dates_loop[i])
        df_1 = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file = r'''file csv\DATA Elia\RES\Wind off shore\WindForecast_'''+str(n_year)+'''-'''+str(n_month)+'''.xls'''
        file = f"{config_parameters['loc_data'][0]}/Elia/RES/Wind off shore/WindForecast_{n_year}-{n_month}.xls"

        if os.path.isfile(file)==True:
                # Load excel file
                file_unprocessed = pd.read_excel(file, header=0, skiprows=3)
                file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['DateTime'], format='%d/%m/%Y %H:%M')
                file_unprocessed.set_index('DateTime', inplace=True)  # column Date/Time as index
                percent_missing = file_unprocessed.isnull().sum() * 100 / len(file_unprocessed)  # percentage of missing data per column
                print('Missing data (%) : DA ' + str(round(percent_missing['Day-ahead forecast (11h00) [MW]'], 2)) + ', RT '\
                      + str(round(percent_missing['Measured & upscaled [MW]'], 2)))
                if percent_missing['Day-ahead forecast (11h00) [MW]'] > 20: # Replace day-ahead forecast by most recent forecast.
                    file_unprocessed['Day-ahead forecast (11h00) [MW]'] = file_unprocessed['Most recent forecast [MW]']
                file_unprocessed = file_unprocessed.resample('15min').mean().interpolate(limit_direction='both')  # Deals with spring hour and fall hour.
                for date_dst_fall in DST_in_fall:
                    if date_dst_fall in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ')  # resample seems to do the job for removing the added hour in fall
                        #print(file_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                for date_dst_spring in DST_in_spring:
                    if date_dst_spring in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ')  # resample seems to do the job for interpolating the missing hour in spring
                        #print(file_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                df_1 = df_1.assign(FROM_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(TO_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(DA_wind_off_shore_MW=file_unprocessed['Day-ahead forecast (11h00) [MW]'].values)
                df_1 = df_1.assign(RT_wind_off_shore_MW=file_unprocessed['Measured & upscaled [MW]'].values)
                df_1 = df_1.assign(Capacity_wind_off_shore_MW=file_unprocessed['Monitored Capacity [MW]'].values)
                df = df._append(df_1, ignore_index=True)



        else:
                print('This file doesn t exist',file)

    df = df.set_index('FROM_DATE')
    print(df.shape)
    print(df.head())
    print(df.tail())
    # Ecritrue fichier CSV
    open(file_xlsx, 'w+')
    df.to_excel(file_xlsx)
    # break
    df.to_hdf(file_h5, 'DATA_ELIA_wind_off_shore')


##############################################################################
#Wind ON shore
########################################################################
def import_quarter_hourly_data_wind_on_shore(config_parameters):
    # Ecritrue fichier CSV
    file_xlsx = r'''file csv\DATA Elia\RES\\Wind on shore\Global_data.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/RES/Wind on shore/Global_data.xlsx"

    file_h5 = config_parameters['file_h5'][0]

    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE', 'DA_wind_on_shore_MW', 'RT_wind_on_shore_MW','Capacity_wind_on_shore_MW'], dtype=float)
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H

    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]
    for i in range(len(dates_loop)):
        print(dates_loop[i])
        df_1 = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file = r'''file csv\DATA Elia\RES\Wind on shore\WindForecast_'''+str(n_year)+'''-'''+str(n_month)+'''.xls'''
        file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/RES/Wind on shore/WindForecast_{n_year}-{n_month}.xls"

        if os.path.isfile(file)==True:
                # Load excel file
                file_unprocessed = pd.read_excel(file, header=0, skiprows=3)
                file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['DateTime'], format='%d/%m/%Y %H:%M')
                file_unprocessed.set_index('DateTime', inplace=True)  # column Date/Time as index

                percent_missing = file_unprocessed.isnull().sum() * 100 / len(file_unprocessed)  # percentage of missing data per column
                print('Missing data (%) : DA ' + str(round(percent_missing['Day-ahead forecast (11h00) [MW]'], 2)) + ', RT '\
                      + str(round(percent_missing['Measured & upscaled [MW]'], 2)))
                if percent_missing['Day-ahead forecast (11h00) [MW]'] > 20: # Replace day-ahead forecast by most recent forecast.
                    file_unprocessed['Day-ahead forecast (11h00) [MW]'] = file_unprocessed['Most recent forecast [MW]']
                file_unprocessed = file_unprocessed.resample('15min').mean().interpolate(limit_direction='both')  # Deals with spring hour and fall hour.
                for date_dst_fall in DST_in_fall:
                    if date_dst_fall in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ')  # resample seems to do the job for removing the added hour in fall
                        #print(file_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                for date_dst_spring in DST_in_spring:
                    if date_dst_spring in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ')  # resample seems to do the job for interpolating the missing hour in spring
                        #print(file_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                df_1 = df_1.assign(FROM_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(TO_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(DA_wind_on_shore_MW=file_unprocessed['Day-ahead forecast (11h00) [MW]'].values)
                df_1 = df_1.assign(RT_wind_on_shore_MW=file_unprocessed['Measured & upscaled [MW]'].values)
                df_1 = df_1.assign(Capacity_wind_on_shore_MW=file_unprocessed['Monitored Capacity [MW]'].values)
                df = df._append(df_1, ignore_index=True)

        else:
            print('This file doesn t exist', file)

    df = df.set_index('FROM_DATE')
    print(df.shape)
    print(df.head())
    print(df.tail())
    open(file_xlsx, 'w+')
    df.to_excel(file_xlsx)
    # break
    df.to_hdf(file_h5, 'DATA_ELIA_wind_on_shore')

##############################################################################
#Wind Total
########################################################################
def import_quarter_hourly_data_wind_total(config_parameters):
    # Ecritrue fichier CSV
    file_xlsx = r'''file csv\DATA Elia\RES\\Wind Total\Global_data.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/RES/Wind Total/Global_data.xlsx"

    file_h5 = config_parameters['file_h5'][0]

    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE', 'DA_wind_total_MW', 'RT_wind_total_MW','Capacity_wind_total_MW'], dtype=float)
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H

    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]
    for i in range(len(dates_loop)):
        print(dates_loop[i])
        df_1 = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file = r'''file csv\DATA Elia\RES\Wind Total\WindForecast_'''+str(n_year)+'''_'''+str(n_month)+'''.xls'''
        file = f"{config_parameters['loc_data'][0]}/Elia/RES/Wind Total/WindForecast_{n_year}_{n_month}.xls"

        if os.path.isfile(file)==True:
                # Load excel file
                file_unprocessed = pd.read_excel(file, header=0, skiprows=3)
                try:
                    file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['DateTime'], format='%d/%m/%Y %H:%M')
                except:
                    file_unprocessed.columns = file_unprocessed.iloc[1]
                    file_unprocessed = file_unprocessed.iloc[2:]
                    file_unprocessed['DateTime'] = pd.to_datetime(file_unprocessed['DateTime'], format='%d/%m/%Y %H:%M')

                file_unprocessed.set_index('DateTime', inplace=True)  # column Date/Time as index

                percent_missing = file_unprocessed.isnull().sum() * 100 / len(file_unprocessed)  # percentage of missing data per column
                print('Missing data (%) : DA ' + str(round(percent_missing['Day-ahead forecast (11h00) [MW]'], 2)) + ', RT '\
                      + str(round(percent_missing['Measured & upscaled [MW]'], 2)))
                if percent_missing['Day-ahead forecast (11h00) [MW]'] > 20: # Replace day-ahead forecast by most recent forecast.
                    file_unprocessed['Day-ahead forecast (11h00) [MW]'] = file_unprocessed['Most recent forecast [MW]']
                fu = file_unprocessed.drop(['Active Decremental Bids [yes/no]'],axis=1)
                fu = fu.apply(pd.to_numeric,errors='coerce')
                file_unprocessed = fu.resample('15min').mean().interpolate(limit_direction='both')  # Deals with spring hour and fall hour.
                for date_dst_fall in DST_in_fall:
                    if date_dst_fall in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ')  # resample seems to do the job for removing the added hour in fall
                        #print(file_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                for date_dst_spring in DST_in_spring:
                    if date_dst_spring in file_unprocessed.index.get_level_values('DateTime'):
                        print('OK ')  # resample seems to do the job for interpolating the missing hour in spring
                        #print(file_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
                df_1 = df_1.assign(FROM_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(TO_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]))
                df_1 = df_1.assign(DA_wind_total_MW=file_unprocessed['Day-ahead forecast (11h00) [MW]'].values)
                df_1 = df_1.assign(RT_wind_total_MW=file_unprocessed['Measured & upscaled [MW]'].values)
                df_1 = df_1.assign(Capacity_wind_total_MW=file_unprocessed['Monitored Capacity [MW]'].values)
                df = df._append(df_1, ignore_index=True)

        else:
            print('This file doesn t exist', file)

    df = df.set_index('FROM_DATE')
    print(df.shape)
    print(df.head())
    print(df.tail())
    open(file_xlsx, 'w+')
    df.to_excel(file_xlsx)
    # break
    df.to_hdf(file_h5, 'DATA_ELIA_wind_total')






if __name__ == '__main__':
    config_parameters = pd.DataFrame({'start_trainingData': [datetime(2014, 1, 1)],
                                      'end_trainingData': [datetime(2024, 2, 29)],
                                      'file_h5': 'data_qh_SI_imbPrice.h5',
                                      })
    import_quarter_hourly_data_pv(config_parameters)
    import_quarter_hourly_data_wind_total(config_parameters)

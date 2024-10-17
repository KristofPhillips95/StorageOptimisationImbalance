import pandas as pd
from datetime import date, time, datetime, timedelta
from dateutil.rrule import rrule, MONTHLY, YEARLY
import numpy as np
import os
# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
# Change directory
os.chdir(cwd)


def import_hourly_data_processing_Day_Ahead_market(config_parameters):

    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE' ], dtype=float)#], dtype=float)
    name_file = ['\Belgium (BE)']#,'\DE-AT-LU','\_Netherlands (NL)','\France (FR)'

    '''
    #Time change in spring
    #DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0)] # - 1 H
    DST_in_spring=["30.03.2014","29.03.2015","27.03.2016","26.03.2017","25.03.2018"]
    #Time change in fall
    #DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0)] # + 1 H
    DST_in_fall=["26.10.2014","25.10.2015","30.10.2016","29.10.2017","28.10.2018"]

    dates_loop = [dt for dt in rrule(YEARLY, dtstart=start_trainingData, until=end_trainingData)]



    for i in range(len(dates_loop)):
        print(dates_loop[i])
        df_1_quarter_hourly = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        df_1_hourly = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        n_year = dates_loop[i].year
        #file = r file csv\DATA entsoe transparency platform\Day Ahead market\Belgium (BE)\Day-ahead Prices_ +str(n_year)+ .xlsx
        if os.path.isfile(file)==True:
            file_unprocessed=pd.read_excel(file, header=None, skiprows=6)
            file_unprocessed.drop([2,3,4,5], axis=1, inplace=True)
            file_unprocessed[1] = file_unprocessed[1].astype(str)
            file_unprocessed[1]=file_unprocessed[1].str.split('\s+').str[0]
            indice_fall = file_unprocessed[file_unprocessed[0].str.contains(DST_in_fall[i],na=False)].index.values
            file_unprocessed.drop(file_unprocessed.index[int(indice_fall + 4)],inplace=True)  ####################### DIFFERENCE with quarter_hourly
            file_unprocessed = file_unprocessed.reset_index(drop=True)
            file_unprocessed = file_unprocessed[pd.notnull(file_unprocessed[0])]
            file_unprocessed = file_unprocessed[~file_unprocessed[0].str.contains('.'+str(n_year))]
            file_processed=file_unprocessed.reset_index(drop = True)
            del file_processed[0]
            file_processed[1] = file_processed[1].astype(str)
            file_processed[1] = file_processed[1].replace({'-': np.nan})
            file_processed[1] = file_processed[1].astype('float')
            percent_missing = file_processed.isnull().sum() * 100 / len(file_processed)  # percentage of missing data per column
            print( 'Missing data (%) : DA-ahead prices_ '+str(round(percent_missing[1],2)) )
            Start_date_from = datetime(int(n_year), 1, 1, 0, 0, 0)
            file_processed = file_processed.assign(FROM_DATE=pd.Series([Start_date_from + timedelta(days=0, hours=i, minutes=0, seconds=0) for i in range(file_processed.shape[0])]))
            file_processed = file_processed.set_index('FROM_DATE')
            file_processed = file_processed.resample('H').mean().interpolate(limit_direction='both')
            file_processed.rename(columns={1: 'Day-ahead Price [€\MWh]'}, inplace=True)

            file_processed = file_processed.resample('15T').pad()# Hourly to quarter-hourly data
            Upsample_last_edge = pd.DataFrame(
                {'FROM_DATE': [file_processed.index[-1]+ timedelta(days=0, hours=0, minutes=15*(i+1), seconds=0) for i in range(3)],
                 'Day-ahead Price [€\MWh]': [file_processed['Day-ahead Price [€\MWh]'].iloc[-1].copy() for _ in range(3)]})
            Upsample_last_edge = Upsample_last_edge.set_index('FROM_DATE')
            file_processed = file_processed.append(Upsample_last_edge)
            df = df.append(file_processed, ignore_index=True,sort=True)

    #df = df.set_index('FROM_DATE')
    #print(df)
    #Ecritrue fichier CSV
    df = df.assign(FROM_DATE=pd.Series([start_trainingData + timedelta(days=0, hours=0, minutes=i*15, seconds=0) for i in range(df.shape[0])]))
    df = df.assign(TO_DATE=pd.Series([start_trainingData + timedelta(days=0, hours=0, minutes=(i+1)*15, seconds=0) for i in range(df.shape[0])]))
    df = df.set_index('FROM_DATE')
    print(df)

    file = r file csv\DATA entsoe transparency platform\Day Ahead market\Belgium (BE)\Global_data.xlsx
    open(file, 'w+')
    df.to_excel(file)
    #break
    file_h5_quarter_hourly = r DATA_SI_prediction_15T.h5
    df.to_hdf(file_h5_quarter_hourly, 'DA_market_prices')
    '''
    #only hourly data
    file_h5 = config_parameters['file_h5'][0]
    DST_in_spring=["30.03.2014","29.03.2015","27.03.2016","26.03.2017","25.03.2018","31.03.2019","29.03.2020","28.03.2021","27.03.2022","26.03.2023"] # - 1 H WHY IS THIS NOT USED ANYWHERE?
    DST_in_fall=["26.10.2014","25.10.2015","30.10.2016","29.10.2017","28.10.2018","27.10.2019","25.10.2020","31.10.2021","30.10.2022","20.10.2023"] # + 1 H

    name_file_h5 = ['BE']#, 'GER', 'NL', 'FR'
    Year = ['2014','2015','2016','2017','2018','2019','2020','2021','2022', '2023']
    for j in range(len(name_file)):
        df=pd.DataFrame(columns=['Day-ahead Price [€\MWh]', 'FROM_DATE'], dtype=float)
        print(j,name_file[j])
        for i in range(len(Year)):
            print(Year[i])
            file = r'''file csv\DATA entsoe transparency platform\Day Ahead market'''+name_file[j]+'''\Day-ahead Prices_''' + str(Year[i]) + '''.xlsx'''
            file = f"{config_parameters['loc_data'][0]}/ENTSOE/Day Ahead market/Belgium (BE)/Day-ahead Prices_{Year[i]}.xlsx"

            # Load excel file
            file_unprocessed=pd.read_excel(file, header=None, skiprows=6, engine = 'openpyxl')
            #file_unprocessed.drop([2,3,4,5], axis=1, inplace=True)
            #print(file_unprocessed)
            file_unprocessed[1] = file_unprocessed[1].astype(str)
            #print(file_unprocessed[1].str.split(' EUR').str[0])
            file_unprocessed[1]=file_unprocessed[1].str.split('\s+').str[0]
            print(file_unprocessed)

            #indice_spring = file_unprocessed[file_unprocessed[0].str.contains(DST_in_spring[i],na=False)].index.values
            #print(indice_spring)
            #print([file_unprocessed.iloc[indice_spring+10+i] for i in range(4)])
            #file_unprocessed.drop(file_unprocessed.index[[int(indice_spring+10), int(indice_spring+11), int(indice_spring+12), int(indice_spring+13)]], inplace=True)
            #file_unprocessed=file_unprocessed.reset_index(drop = True)
            print(DST_in_fall[i])
            indice_fall = file_unprocessed[file_unprocessed[0].str.contains(DST_in_fall[i],na=False)].index.values
            print(indice_fall)
            print(file_unprocessed.iloc[indice_fall+4])
            file_unprocessed.drop(file_unprocessed.index[int(indice_fall[0]+4)], inplace=True)   ####################### DIFFERENCE with quarter_hourly
            file_unprocessed=file_unprocessed.reset_index(drop = True)

            #print(file_unprocessed[pd.isnull(file_unprocessed[0])])
            file_unprocessed = file_unprocessed[pd.notnull(file_unprocessed[0])]
            #print(file_unprocessed[file_unprocessed[0].str.contains('.'+str(Year[i]))])
            #print(file_unprocessed.iloc[indice_fall:indice_fall])
            file_unprocessed = file_unprocessed[~file_unprocessed[0].str.contains('.'+str(Year[i]))]
            file_processed=file_unprocessed.reset_index(drop = True)
            del file_processed[0]

            Start_date_from = datetime(int(Year[i]), 1, 1, 0, 0, 0)
            file_processed = file_processed.assign(FROM_DATE=pd.Series([Start_date_from+timedelta(days=0, hours=i, minutes=0, seconds=0) for i in range(file_processed.shape[0]) ]))
            file_processed.rename(columns={1:'Day-ahead Price [€\MWh]'}, inplace=True)
            df = df.append(file_processed,ignore_index=True)
            #break
        df['Day-ahead Price [€\MWh]']=df['Day-ahead Price [€\MWh]'].astype(str)
        df['Day-ahead Price [€\MWh]'] = df['Day-ahead Price [€\MWh]'].replace({'-': np.nan})
        df['Day-ahead Price [€\MWh]'] =  df['Day-ahead Price [€\MWh]'].astype('float')
        df = df.set_index('FROM_DATE')
        print(df)
        print(df.shape)
        print(df.columns)
        print(df.head())
        print(df.tail())


        percent_missing = df.isnull().sum() * 100 / len(df)  # percentage of missing data per column
        print('Missing data (%) : DA Prices ' + str(round(percent_missing['Day-ahead Price [€\MWh]'], 2)))

        df = df.resample('H').mean().interpolate(limit_direction='both')  # Deals with spring hour and fall hour.

        percent_missing = df.isnull().sum() * 100 / len(df)  # percentage of missing data per column
        print('Missing data (%) : DA Prices ' + str(round(percent_missing['Day-ahead Price [€\MWh]'], 2)))

        file_xlsx = r'''file csv\DATA entsoe transparency platform\Day Ahead market\Belgium (BE)\Global_data.xlsx'''
        file_xlsx = f"{config_parameters['loc_data'][0]}/ENTSOE/Day Ahead market/Belgium (BE)/Global_data.xlsx"
        #file_xlsx = r'''file csv\DATA Elia\RES\\Wind Total\Global_data.xlsx'''

        #Ecritrue fichier xlsx
        open(file_xlsx, 'w+')
        df.to_excel(file_xlsx,engine='xlsxwriter')
        df.to_hdf(file_h5, 'DATA_DA_Prices')

if __name__ == '__main__':
    config_parameters = pd.DataFrame({'start_trainingData': [datetime(2014, 1, 1)],
                                      'end_trainingData': [datetime(2024, 1, 1)],
                                      'file_h5': 'data_qh_SI_imbPrice.h5',
                                      })
    import_hourly_data_processing_Day_Ahead_market(config_parameters)
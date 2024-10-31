# Import pandas
import pandas as pd
from datetime import date, time, datetime, timedelta
import os
import math
from dateutil.rrule import rrule, MONTHLY, YEARLY
# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
# Change directory
os.chdir(cwd)


####################################
def import_quarter_hourly_data_imbalancenrvprices(config_parameters):
    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    df = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE'], dtype=float) # Instantiating a dataframe to save data over the excel files
    #file_xlsx =r'''file csv\DATA Elia\imbalancenrvprices\Global_data.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/imbalancenrvprices/Global_data.xlsx"
    file_h5 = config_parameters['file_h5'][0]

    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H



    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]
    for i in range(len(dates_loop)):
        print(dates_loop[i])
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file_R2 = r'''file csv\DATA Elia\imbalancenrvprices\Imbalance-'''+str(n_year)+'''-'''+str(n_month)+'''.xls'''
        file_R2 = f"{config_parameters['loc_data'][0]}/Elia/imbalancenrvprices/Imbalance-{n_year}-{n_month}.xls"

        if os.path.isfile(file_R2) == True:
            file_unprocessed = pd.read_excel(file_R2, header=0, skiprows=1)
            file_unprocessed["period"] = file_unprocessed["Date"]+" "+file_unprocessed["Quarter"].str.split(' ').str[0] # create a timestamp in a correct format
            file_unprocessed.drop(['Status','Date','Quarter'], axis=1, inplace=True) # remove useless columns
            if 'SR\n (€/MWh)' in file_unprocessed.columns:
                file_unprocessed.drop(['SR\n (€/MWh)'], axis=1, inplace=True)
            file_unprocessed['period']=pd.to_datetime(file_unprocessed['period'], format='%d/%m/%Y %H:%M') # column period in datetime format
            file_unprocessed.set_index('period',inplace=True) # column Date/Time as index
            file_unprocessed['ACE'] =  file_unprocessed['NRV\n(MW)']  + file_unprocessed['SI\n (MW)']
            #old:
            file_unprocessed['SI'] = file_unprocessed['SI\n (MW)']
            file_unprocessed['Imb_price'] = file_unprocessed['POS\n (€/MWh)']
            percent_missing = file_unprocessed.isnull().sum() * 100 / len(file_unprocessed) # percentage of missing data per column
            print( 'Missing data (%) : NRV '+str(round(percent_missing['NRV\n(MW)'],2))+\
                   ', SI '+str(round(percent_missing['SI\n (MW)'],2))+\
                   ', ACE '+str(round(percent_missing['ACE'],2)) +\
                   ', Imb_price '+str(round(percent_missing['Imb_price'],2))
                   )
            file_unprocessed = file_unprocessed.resample('15min').mean().interpolate(limit_direction='both') # Deals with spring hour and fall hour.
            for date_dst_fall in DST_in_fall:
                if date_dst_fall in file_unprocessed.index.get_level_values('period'):
                    print('OK ') # resample seems to do the job for removing the added hour in fall
                    #print(file_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
            for date_dst_spring in DST_in_spring:
                if date_dst_spring in file_unprocessed.index.get_level_values('period'):
                    print('OK ') # resample seems to do the job for interpolating the missing hour in spring
                    #print(file_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
            file_unprocessed.rename(columns={'NRV\n(MW)':'NRV [MW]','SI\n (MW)':'SI [MW]','a\n (€/MWh)':'alpha [€/MWh]','MIP\n (€/MWh)':'MIP [€/MWh]','MDP\n (€/MWh)':'MDP [€/MWh]',
                                               'POS\n (€/MWh)':'POS [€/MWh]','NEG\n (€/MWh)':'NEG [€/MWh]'}, inplace=True)
            file_unprocessed.drop(['NRV [MW]','SI [MW]','MIP [€/MWh]','MDP [€/MWh]',\
                                   'POS [€/MWh]','NEG [€/MWh]'], axis=1,inplace=True)  # removing useless columns
            file_unprocessed['FROM_DATE'] = [ dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]
            file_unprocessed["TO_DATE"] = [dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_unprocessed.shape[0])]
            df = pd.concat([df,file_unprocessed], ignore_index=True,sort=True)
        else:
            print('This file doesn t exist',file_R2)
    df = df.set_index('FROM_DATE')
    print(df.shape)
    print(df)
    # Ecritrue fichier CSV
    open(file_xlsx, 'w+')
    df.to_excel(file_xlsx)
    # break
    df.to_hdf(file_h5, 'DATA_ELIA_imbalancenrvprices')




if __name__ == '__main__':
    config_parameters = pd.DataFrame({'start_trainingData': [datetime(2014, 1, 1)],
                                      'end_trainingData': [datetime(2024, 2, 29)],
                                      #old: 'file_h5': 'DATA_ACE_15T.h5',
                                      'file_h5': 'data_qh_SI_imbPrice.h5',
                                      })
    import_quarter_hourly_data_imbalancenrvprices(config_parameters)

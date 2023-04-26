import datetime as dt
import elia
import time

###################
#Helper Methods
###################

def get_start_and_end_time_data(attr):
    if attr in ["SI","CBF"]:
        delta = dt.timedelta(hours=5)
        end = dt.datetime.now()
        start = end -delta
    elif attr in ["DA_FT","load"]:
        delta = dt.timedelta(hours=24)
        end = dt.datetime.now()
        start = end - delta
    elif attr in ["RES"]:
        delta = dt.timedelta(hours=72)
        end = dt.datetime.now() + dt.timedelta(hours=24)
        start = end - delta
    elif attr in ["ARC"]:
        delta = dt.timedelta(hours=72)
        end = dt.datetime.now() + dt.timedelta(hours=48)
        start = end - delta
    else:
        raise ValueError(f"Start and end time not configured for {attr}")
    return start, end

######################
#Create connection object
######################
connection = elia.EliaPandasClient()


########################################
# Cross border flows
########################################
print("Starting timer for border_flows")
timer = time.time()
start,end = get_start_and_end_time_data("CBF")
df_CB = connection.get_cross_border_flows_per_quarter_hour(start = start,end=end)
print(f"Time spent for CB equals {time.time()-timer}")

####################################################
#Conventional generation forecast by fuel type
#####################################################
start,end = get_start_and_end_time_data("DA_FT")
df_DA_FT = connection.get_DA_schedule_by_fuel()

####################################################
#System imbalance information
#####################################################
start,end = get_start_and_end_time_data("SI")
# df_SI_quarter = connection.get_imbalance_prices_per_quarter_hour(start=start,end=end)
df_SI_quarter = connection.get_imbalance_prices_per_quarter_hour_own(start=start,end=end)

df_SI_min = connection.get_imbalance_prices_per_min()
####################################################
#System load
#####################################################
start,end = get_start_and_end_time_data("load")

df_load = connection.get_load_on_elia_grid(start=start, end=end)
df_load = connection.get_load_on_elia_grid(start=start, end=end)
####################################################
#RES Forecast
#####################################################
start,end = get_start_and_end_time_data("RES")
#Note: these NRT datasets only go back about half a day I think, but it should be possible to use another dataset with historical values to go back further
df_wind = connection.get_historical_wind_power_estimation_and_forecast_own(start=start,end=end)
df_solar = connection.get_historical_solar_power_estimation_and_forecast_own(start=start,end=end)

####################################################
#DA Prices
#####################################################

##TODO

####################################################
#Merit order
#####################################################
start,end = get_start_and_end_time_data("ARC")
df_merit_decr = connection.get_merit_order_decremental(start=start,end=end)
df_merit_incr = connection.get_merit_order_decremental(start=start,end=end)

df_solar = connection.get_historical_solar_power_estimation_and_forecast_own(start=start,end=end)





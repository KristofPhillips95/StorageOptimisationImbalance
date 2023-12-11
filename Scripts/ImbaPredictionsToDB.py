import requests
import json
import datetime
import predict_imb_price

#Establish ID of object to be pushed to DB
index = 3

#API link to push to DB
api_link = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items"
api_link_2 = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/lts_items"
#Initial soc
soc_0 = 2

# #Obtain relevant values from the forecaster
# si_quantile_fc, avg_price_fc, quantile_price_fc, quantiles, curr_qh,\
# (last_si_value,last_si_time), (last_imbPrice_value,last_imbPrice_dt), (c,d,soc)\
#     = predict_imb_price.call_prediction(soc_0)
#
# #Establish a list of future times for which the forecasts are made
# fc_times = [(curr_qh + datetime.timedelta(minutes=15 * (fc_step+1))).strftime("%d %H:%M:%S") for fc_step in
#                 range(si_quantile_fc.shape[0])]
#
# #Establish a list of times for which values are unknown
# unknown_times = [(last_si_time + datetime.timedelta(minutes=15 * (fc_step+1))).strftime("%d %H:%M:%S") for fc_step in
#                 range(si_quantile_fc.shape[0]) if (last_si_time + datetime.timedelta(minutes=15 * (fc_step+1)))<=curr_qh]
#
# #Convert the 2d imbalance forecast array to a dictionary with timesteps as keys
# si_quantile_fc_dict = dict()
# for i, fc_time in enumerate(fc_times):
#     si_quantile_fc_dict[fc_time] = [str(si_quantile_fc[i, q]) for q in range(si_quantile_fc.shape[1])]
#
# quantile_price_fc_dict = dict()
# for i, fc_time in enumerate(fc_times):
#     quantile_price_fc_dict[fc_time] = [str(quantile_price_fc[i, q]) for q in range(quantile_price_fc.shape[1])]

writing_time = datetime.datetime.now()
# data = {
#     "id": index,
#     "curr_qh":curr_qh.strftime('%d %H:%M:%S'),
#     "last_imbPrice_value": last_imbPrice_value,
#     "last_imbPrice_dt": last_imbPrice_dt.strftime('%d %H:%M:%S'),
#     "last_si_value": last_si_value,
#     "last_si_time":last_si_time.strftime('%d %H:%M:%S'),
#     "charge": c.tolist(),
#     "discharge":d.tolist(),
#     "soc": soc.tolist(),
#     "quantiles": quantiles,
#     "si_quantile_fc": si_quantile_fc_dict,
#     "avg_price_fc": avg_price_fc.tolist(),
#     "quantile_price_fc":quantile_price_fc_dict,
#     "writing_time": writing_time.strftime('%d %H:%M:%S'),
#     "unkown_times":unknown_times
# }
total_profit = 1
data_2 = {
    "id": index,
    "writing_time": writing_time.strftime('%d %H:%M:%S'),
    "time": writing_time.strftime('%d %H:%M:%S'),
    "soc": 1,
    "charge": 1,
    "discharge": 0,
    "total_profit": total_profit,
}

print(f"Writing now to API:",writing_time)

#response = requests.put(api_link, json=data)
response = requests.put(api_link_2, json=data_2)
print(response.text)
import requests
import json
import datetime
import predict_imb_price

SI_FC, last_si, quantiles = predict_imb_price.pred_SI(dev='cpu')

id = 11
api_link = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items"

prices_fc_spread = dict()

prices_fc_spread = dict()
last_si_time = last_si[1]
fc_times = [(last_si_time + datetime.timedelta(minutes=15 * fc_step)).strftime("%H:%M:%S") for fc_step in range(8)]

for i, fc_time in enumerate(fc_times):
    prices_fc_spread[fc_time] = [str(SI_FC[i,q]) for q in range(SI_FC.shape[1]) ]


data = {
    "id": id,
    "time": last_si_time.strftime('%Y-%m-%d %H:%M:%S'),
    "imba_price": last_si[0],
    "imba_price_fc": 1,
    "charge": 1,
    "soc": 1,
    "fc_spread": prices_fc_spread,
    "quantiles": quantiles
}

# data = {
#     "id": id * j,
#     "time": some_time_str,
#     "imba_price": imba_price,
#     "imba_price_fc": imba_price_fc,
#     "charge": charge,
#     "soc": soc,
#     "fc_spread": prices_fc_spread
# }

print(f"Writing now to API:",last_si_time)

response = requests.put(api_link, json=data)
print(response.text)
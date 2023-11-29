import datetime
import pandas as pd
import os
import requests
import numpy as np

ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
prices = [0, 100, 50, 40, 200, 100, 20, 0, 100, 100]
prices_fc = [10, 90, 100, 30, 180, 200, 80, 0, 40, 0]

median_fc = np.median(prices_fc)
charges = [1 if fc < median_fc else -1 for fc in prices_fc]

socs = [5]
soc = 5

time = datetime.datetime.now()
time_str = time.strftime("%H:%M:%S")

for charge in charges:
    soc = soc + charge
    socs.append(soc)


for j in range(1, 5, 1):
    for id, imba_price, imba_price_fc, charge, soc in zip(ids, prices, prices_fc, charges, socs):
        some_time = time + datetime.timedelta(minutes=15 * id)
        some_time_str = some_time.strftime("%H:%M:%S")
        prices_fc_spread = dict()
        fc_times = [(some_time + datetime.timedelta(minutes=15 * fc_step)).strftime("%H:%M:%S") for fc_step in range(8)]

        for i, fc_time in enumerate(fc_times):
            prices_fc_spread[fc_time] = [imba_price * 0.9 ** i, imba_price * 1.1 ** i]

        data = {
            "id": id * j,
            "time": some_time_str,
            "imba_price": imba_price,
            "imba_price_fc": imba_price_fc,
            "charge": charge,
            "soc": soc,
            "fc_spread": prices_fc_spread
        }

        # print(f"Testing API get", current_time)
        # #
        # # response = requests.get("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items")
        # # print(response.text)
        #
        print(f"Writing now to API:", some_time_str)

        response = requests.put("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items", json=data)
        print(response.text)

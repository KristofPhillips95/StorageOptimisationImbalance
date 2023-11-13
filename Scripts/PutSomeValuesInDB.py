from datetime import datetime
import pandas as pd
import os
import requests
import numpy as np

now = datetime.now()
current_time = now.strftime("%H:%M:%S")


ids = [1,2,3,4,5,6,7,8,9,10]
prices = [0,100,50,40,600,100,20,0,100,0]
prices_fc = [10,90,100,30,500,200,80,0,40,0]
prices_fc_spread = [[fc*0.9,fc*1.1] for fc in prices_fc]
median_fc = np.median(prices_fc)
charges = [1 if fc< median_fc else -1 for fc in prices_fc]

socs = [5]
soc = 5
for charge in charges:
    socs.append(soc)
    soc = soc + charge
for j in range(1,2,1):
    for id,imba_price,imba_price_fc,charge,soc,fc_spread in zip(ids,prices,prices_fc,charges,socs,prices_fc_spread):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        data = {
            "id": id*j,
            "time": current_time,
            "imba_price": imba_price,
            "imba_price_fc":imba_price_fc,
            "charge": charge,
            "soc": soc,
            "fc_spread":fc_spread
        }

        # print(f"Testing API get", current_time)
        # #
        # # response = requests.get("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items")
        # # print(response.text)
        #
        print(f"Writing now to API:", current_time)

        response = requests.put("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items", json=data)
        print(response.text)

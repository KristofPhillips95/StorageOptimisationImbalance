from datetime import datetime
import pandas as pd
import os
import requests
import numpy as np

now = datetime.now()
current_time = now.strftime("%H:%M:%S")


ids = [1,2,3,4,5,6,7,8,9,10]
prices = [0,100,50,40,600,100,20,0,100,0]
prices_fc = [10,90,40,40,500,200,80,0,40,0]
median_fc = np.median(prices_fc)
charges = [1 if fc< median_fc else -1 for fc in prices_fc]

for id,imba_price,imba_price_fc,charge in zip(ids,prices,prices_fc,charges):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    data = {
        "id": id,
        "time": current_time,
        "imba_price": imba_price,
        "imba_price_fc":imba_price_fc,
        "charge": charge
    }

    print(f"Testing API get", current_time)

    response = requests.get("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items")
    print(response.text)

    print(f"Writing now to API:", current_time)

    response = requests.put("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items", json=data)
    print(response.text)

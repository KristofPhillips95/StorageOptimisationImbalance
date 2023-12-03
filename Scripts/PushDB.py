import datetime
# import pandas as pd
# import os
import requests
from predict_imb_price import pred_SI
import time

def write_item_API_and_reschedule(scheduler, interval=60,index=0):
    # Schedule the next call first
    scheduler.enter(interval, 1, write_item_API_and_reschedule, (scheduler, interval,index+1,))
    # Then do your stuff
    write_item_API(index)

def try_creating_item(index):
    SI_FC, last_si, quantiles = pred_SI(dev='cpu')
    prices_fc_spread = dict()
    last_si_time = last_si[1] + datetime.timedelta(minutes=60)
    fc_times = [(last_si_time + datetime.timedelta(minutes=15 * (fc_step+1))).strftime("%H:%M:%S") for fc_step in
                range(SI_FC.shape[0])]
    print(SI_FC.shape)
    writing_time = datetime.datetime.now().strftime('%H:%M:%S')
    for i, fc_time in enumerate(fc_times):
        prices_fc_spread[fc_time] = [str(SI_FC[i, q]) for q in range(SI_FC.shape[1])]
    data = {
        "id": index,
        "time": last_si_time.strftime('%H:%M:%S'),
        "imba_price": last_si[0],
        "imba_price_fc": 1,
        "charge": 1,
        "soc": 1,
        "fc_spread": prices_fc_spread,
        "quantiles": quantiles,
        "writing time": writing_time

    }
    index+=1
    return data


def write_item_API(index, max_retries=7, current_retry=0):
    """
    Attempt to create and write an item to an API with retry mechanism.

    Parameters:
    - index (int): The index for creating the item.
    - max_retries (int): The maximum number of retry attempts (default is 5).
    - current_retry (int): The current retry attempt (default is 0).

    Raises:
    - Exception: If an error occurs during item creation or API writing.

    Notes:
    - The function makes use of try_creating_item to attempt item creation.
    - It sends a PUT request to "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items".
    - If an exception occurs, the function prints an error message, waits for 2 minutes, and retries.
    - The retry mechanism continues until the maximum number of retries is reached.

    Example:
     write_item_API(index=123, max_retries=3)
    """
    print(index)
    try:
        data = try_creating_item(index=index)
        print(f"Writing now to API:", data["writing time"])
        response = requests.put("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items", json=data)
        print(response.text)
    except Exception as e:
        print(f"Not writing to API:", e)
        current_retry += 1
        if current_retry <= max_retries:
            #Wait 2 mins before retrying
            time.sleep(120)
            print(f"Retrying (attempt {current_retry}) to create and write item")
            write_item_API(index=index, max_retries=max_retries, current_retry=current_retry)
        else:
            print(f"Maximum retries reached. Exiting.")



    # filename = os.path.join("..", "time_file.csv")
    # row = pd.DataFrame(data=[current_time], columns=["Timestamp"])
    #
    # if os.path.isfile(filename):
    #     df_times = pd.read_csv(filename, index_col=0)
    #     df_times = pd.concat([df_times, row], ignore_index=True)
    # else:
    #     df_times = row
    # df_times.to_csv(filename)
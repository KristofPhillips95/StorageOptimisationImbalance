import datetime
import requests
import predict_imb_price
import time
from lts_data_handling import process_lts
time_format = "%Y/%m/%d, %H:%M:%S"

api_link = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items"

prev_soc = 2
def write_item_API_and_reschedule(scheduler, interval=60,index=0):
    # Schedule the next call first
    scheduler.enter(interval, 1, write_item_API_and_reschedule, (scheduler, interval,index+1))
    # Then do your stuff
    socs_returned = write_item_API(index)

def try_creating_item(index,prev_soc):
    # Obtain relevant values from the forecaster
    si_quantile_fc, avg_price_fc, quantile_price_fc, quantiles, curr_qh, \
    (last_si_value, last_si_time), (last_imbPrice_value, last_imbPrice_dt), (c, d, soc) \
        = predict_imb_price.call_prediction(prev_soc)

    # Establish a list of future times for which the forecasts are made
    fc_times = [(curr_qh + datetime.timedelta(minutes=15 * (fc_step + 1))).strftime("%d %H:%M:%S") for fc_step in
                range(si_quantile_fc.shape[0])]

    # Establish a list of times for which values are unknown
    unknown_times = [(last_si_time + datetime.timedelta(minutes=15 * (fc_step + 1))).strftime("%d %H:%M:%S") for fc_step
                     in range(si_quantile_fc.shape[0]) if
                     (last_si_time + datetime.timedelta(minutes=15 * (fc_step + 1))) <= curr_qh]

    # Convert the 2d imbalance forecast array to a dictionary with timesteps as keys
    si_quantile_fc_dict = dict()
    for i, fc_time in enumerate(fc_times):
        si_quantile_fc_dict[fc_time] = [str(si_quantile_fc[i, q]) for q in range(si_quantile_fc.shape[1])]

    quantile_price_fc_dict = dict()
    for i, fc_time in enumerate(fc_times):
        quantile_price_fc_dict[fc_time] = [str(quantile_price_fc[i, q]) for q in range(quantile_price_fc.shape[1])]

    writing_time = datetime.datetime.now()
    data = {
        "id": index,
        "curr_qh": curr_qh.strftime(time_format),
        "last_imbPrice_value": last_imbPrice_value,
        "last_imbPrice_dt": last_imbPrice_dt.strftime(time_format),
        "last_si_value": last_si_value,
        "last_si_time": last_si_time.strftime(time_format),
        "charge": c.tolist(),
        "discharge": d.tolist(),
        "soc": soc.tolist(),
        "quantiles": quantiles,
        "si_quantile_fc": si_quantile_fc_dict,
        "avg_price_fc": avg_price_fc.tolist(),
        "quantile_price_fc": quantile_price_fc_dict,
        "writing_time": writing_time.strftime(time_format),
        "unkown_times": unknown_times
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
    global prev_soc
    print(index)
    try:
        data = try_creating_item(index=index,prev_soc=prev_soc)
        print(f"Writing now to API:", data["writing_time"])
        response = requests.put(api_link, json=data)
        print(response.text)
        prev_soc = data["soc"][0]
        process_lts(data)
    except Exception as e:
        print(f"Not writing to API:", e)
        current_retry += 1
        if current_retry <= max_retries:
            #Wait 2 mins before retrying
            time.sleep(60)
            print(f"Retrying (attempt {current_retry}) to create and write item")
            write_item_API(index=index, max_retries=max_retries, current_retry=current_retry)
        else:
            print(f"Maximum retries reached. Exiting.")


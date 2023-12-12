import pickle
from datetime import datetime,timezone
import requests

time_format = "%Y/%m/%d, %H:%M:%S"
api_link_lts = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/lts_items"
def create_data_lts(new_known_time,new_known_price,stored_items,writing_time):
    data_lts = {
        "id": new_known_time.strftime(time_format),
        "writing_time": writing_time,
        "soc": stored_items[new_known_time]["soc"],
        "charge": stored_items[new_known_time]["charge"],
        "discharge": stored_items[new_known_time]["discharge"],
        "Imbalance price":new_known_price,
    }
    return data_lts

def process_lts(data):
    stored_items = read_stored_items()
    new_known_times = get_new_known_times(data,stored_items)
    new_unkown_time = get_new_unknown_time(data)
    add_unknown_to_stored_items(new_unkown_time,data,stored_items)

    for nkt in new_known_times:
        if nkt.strftime(time_format) == data["last_si_time"]:
            price = data["last_imbPrice_value"]
        else:
            price = get_imba_price(nkt)
        lts_data = create_data_lts(nkt,price,stored_items,data["writing_time"])
        write_to_lts_db(lts_data,api_link_lts)
        remove_known_from_stored_items(nkt,stored_items)



def get_new_known_times(data,stored_items):
    last_si_time = datetime.strptime(data["last_si_time"], time_format).replace(tzinfo=timezone.utc)

    new_known_times = [key for key in stored_items if key <= last_si_time]
    return new_known_times

def get_new_unknown_time(data):
    # We will assume now that only one qh, the curr qh returned by the forecaster function is a new unknown qh
    return datetime.strptime(data["curr_qh"], time_format).replace(tzinfo=timezone.utc)



def get_imba_price(ts):
    # TODO fetch price from Elia API
    return 1

def write_to_lts_db(data,api_link):
    print(f"Writing now to API:", data["writing_time"])
    response = requests.put(api_link, json=data)
    print(response.text)

def remove_known_from_stored_items(new_known_time,stored_items):
    del stored_items[new_known_time]
    pass

def add_unknown_to_stored_items(new_unknown_time,data,stored_items):
    stored_items[new_unknown_time] = dict()
    stored_items[new_unknown_time]["charge"] = data["charge"][0]
    stored_items[new_unknown_time]["discharge"] = data["discharge"][0]
    stored_items[new_unknown_time]["soc"] = data["soc"][0]
    pass

def write_stored_items(stored_items):
    with open('stored_results/storeditems.pkl', 'wb') as f:
        pickle.dump(stored_items, f)

def read_stored_items():
    with open('stored_results/storeditems.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict
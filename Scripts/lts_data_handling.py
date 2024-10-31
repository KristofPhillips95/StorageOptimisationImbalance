import pickle
from datetime import datetime,timezone
import datetime as dt
import requests
import Data_Elia_API as dea

time_format = "%Y/%m/%d, %H:%M:%S"
api_link_lts = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/lts_items"
STORED_ITEMS_FILE_PATH = 'stored_results/storeditems.pkl'


def create_data_lts(new_known_time,new_known_price,new_known_SI,stored_items,writing_time):
    """
        Create a dictionary with LTS (Lifetime Statistics) data based on provided parameters.

        Parameters:
        - new_known_time (datetime): The timestamp for the new data point.
        - new_known_price (float): The known price associated with the new data point.
        - stored_items (dict): A dictionary containing stored information for different time points.
        - writing_time (datetime): The timestamp representing when the data is written.

        Returns:
        dict: A dictionary containing LTS data for the given time point.
        """
    # Initialize default values
    soc = 0
    charge = 0
    discharge = 0
    prev_total_d_rev = 0  # Initialize to 0 by default

    # Find the last available time in stored_items
    last_available_time = max(stored_items.keys()) if stored_items else None

    if new_known_time in stored_items:
        # If new_known_time is found, retrieve values from stored_items
        soc = stored_items[new_known_time]["soc"]
        charge = stored_items[new_known_time]["charge"]
        discharge = stored_items[new_known_time]["discharge"]
        prev_total_d_rev = stored_items[new_known_time]["prev_total_d_rev"]
        prev_total_c_cost = stored_items[new_known_time]["prev_total_c_cost"]
    elif last_available_time is not None:
        # If new_known_time is not found, use values from the last available time
        prev_total_d_rev = stored_items[last_available_time]["prev_total_d_rev"]
        prev_total_c_cost = stored_items[last_available_time]["prev_total_c_cost"]

    new_total_d_rev = prev_total_d_rev+ new_known_price * discharge
    new_total_c_cost = prev_total_c_cost + new_known_price * charge
    data_lts = {
        "id": new_known_time.strftime(time_format),
        "writing_time": writing_time,
        "soc": soc,
        "charge":charge ,
        "discharge": discharge,
        "Price":new_known_price,
        "SI": new_known_SI,
        "Total_d_rev":new_total_d_rev,
        "Total_c_cost": new_total_c_cost
    }
    return data_lts


def add_prev_total_costs_to_stored(lts_data,new_unknown_time,stored_items):
    stored_items[new_unknown_time]["prev_total_d_rev"] = lts_data["Total_d_rev"]
    stored_items[new_unknown_time]["prev_total_c_cost"] = lts_data["Total_c_cost"]
    print("Updated total revenue to: ",lts_data["Total_d_rev"])


def process_lts(data):
    stored_items = read_stored_items()
    new_known_times = get_new_known_times(data,stored_items)
    new_unkown_time = get_new_unknown_time(data)

    for nkt in new_known_times:
        print("Handling new known time: ", nkt)
        if nkt.strftime(time_format) == data["last_si_time"]:
            price = data["last_imbPrice_value"]
            si = data["last_si_value"]
        else:
            price = get_imba_price(nkt)
            # si = get_imba(nkt)
            si = 0

        lts_data = create_data_lts(nkt,price,si,stored_items,data["writing_time"])
        write_to_lts_db(lts_data,api_link_lts)
        remove_known_from_stored_items(nkt,stored_items)
    add_unknown_to_stored_items(new_unkown_time,data,stored_items)
    add_prev_total_costs_to_stored(lts_data,new_unkown_time,stored_items=stored_items)
    write_stored_items(stored_items)

def get_new_known_times(data,stored_items):
    last_si_time = datetime.strptime(data["last_si_time"], time_format).replace(tzinfo=timezone.utc)

    new_known_times = [key for key in stored_items if key <= last_si_time]
    return new_known_times

def get_new_unknown_time(data):
    # We will assume now that only one qh, the curr qh returned by the forecaster function is a new unknown qh
    return datetime.strptime(data["curr_qh"], time_format).replace(tzinfo=timezone.utc)

def get_imba_price(ts):
    # before = ts - dt.timedelta(minutes=15)
    # after = ts + dt.timedelta(minutes=15)
    # frame_price = dea.get_specific_df(datapoint="SI_and_price",start=before,end=after)
    #
    # return frame_price[frame_price["datetime"] == ts]["positiveimbalanceprice"].item()
    return 0

# def get_imba(ts):
#     # TODO fetch imba from Elia API
#     before = ts - dt.timedelta(minutes=15)
#     after = ts + dt.timedelta(minutes=15)
#     frame_price = dea.get_specific_df(datapoint="SI_and_price",start=before,end=after)
#
#     return frame_price[frame_price["datetime"] == ts]["positiveimbalanceprice"].item()

def write_to_lts_db(data,api_link):
    print(f"Writing to lts DB:", data["writing_time"])
    response = requests.put(api_link, json=data)
    print(response.text)

def remove_known_from_stored_items(new_known_time,stored_items):
    print("removing from locally stored items:", new_known_time)
    del stored_items[new_known_time]
    pass

def add_unknown_to_stored_items(new_unknown_time,data,stored_items):
    print("Adding to locally stored items:", new_unknown_time)

    stored_items[new_unknown_time] = dict()
    stored_items[new_unknown_time]["charge"] = data["charge"][0]
    stored_items[new_unknown_time]["discharge"] = data["discharge"][0]
    stored_items[new_unknown_time]["soc"] = data["soc"][0]

def write_stored_items(stored_items):
    with open(STORED_ITEMS_FILE_PATH, 'wb') as f:
        pickle.dump(stored_items, f)

def read_stored_items():
    with open(STORED_ITEMS_FILE_PATH, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict
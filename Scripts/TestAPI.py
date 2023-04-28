import datetime as dt
import elia

connection = elia.EliaPandasClient()
start = dt.datetime(2022, 11, 20)
end = dt.datetime(2022, 11, 22)

df = connection.get_imbalance_prices_per_quarter_hour(start=start, end=end)
df = connection.get_imbalance_prices_per_quarter_hour(start=start, end=end)
import sched
import time
from PushDB import write_item_API_and_reschedule
import datetime

# Function to calculate the time until the next full quarter hour
def time_until_next_quarter_hour():
    now = datetime.datetime.now()
    minutes_until_quarter_hour = (15 - (now.minute % 15)) % 15
    seconds_until_quarter_hour = 60 - now.second
    time_until_next_quarter_hour = minutes_until_quarter_hour * 60 + seconds_until_quarter_hour
    return time_until_next_quarter_hour

# Calculate the time until the next quarter hour
delay_until_next_quarter_hour = time_until_next_quarter_hour()
print("delay_until_next_quarter_hour" , delay_until_next_quarter_hour)
# Configure the scheduler
interval = 60 * 15
start_index = 1
my_scheduler = sched.scheduler(time.time, time.sleep)

# Log the start of the script
print("Starting the script")

# Schedule the first execution at 10 minutes past the next full quarter hour
my_scheduler.enter(delay_until_next_quarter_hour + 10 * 60, 1, write_item_API_and_reschedule, (my_scheduler, interval, start_index))
#my_scheduler.enter(0, 1, write_item_API_and_reschedule, (my_scheduler, interval, start_index))

# Run the scheduler
my_scheduler.run()

# Log when the script completes
print("Script completed")

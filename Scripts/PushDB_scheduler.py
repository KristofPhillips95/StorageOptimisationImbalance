import sched, time
from PushDB import write_item_API_and_reschedule

interval = 30*15
start_index = 1
my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(interval, 1, write_item_API_and_reschedule, (my_scheduler,interval,start_index))
my_scheduler.run()

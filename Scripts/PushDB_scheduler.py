import sched, time
from PushDB import write_item_API_and_reschedule

interval = 60
my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(interval, 25, write_item_API_and_reschedule, (my_scheduler,interval,))
my_scheduler.run()

# !/usr/bin/env python3
import sched, time
from TimeWriter import write_time_API_and_reschedule

interval = 10
my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(interval, 1, write_time_API_and_reschedule, (my_scheduler,interval,))
my_scheduler.run()


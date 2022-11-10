#!/usr/bin/env python3
import sched, time
import TimeWriter
s = sched.scheduler(time.time, time.sleep)
def do_something(sc): 
    print("Writing Time...")
    # do your stuff
    TimeWriter.write_time()
    sc.enter(10, 1, do_something, (sc,))

s.enter(10, 1, do_something, (s,))
s.run()
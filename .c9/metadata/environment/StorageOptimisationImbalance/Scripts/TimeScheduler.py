{"filter":false,"title":"TimeScheduler.py","tooltip":"/StorageOptimisationImbalance/Scripts/TimeScheduler.py","undoManager":{"mark":24,"position":24,"stack":[[{"start":{"row":0,"column":0},"end":{"row":8,"column":7},"action":"insert","lines":["import sched, time","s = sched.scheduler(time.time, time.sleep)","def do_something(sc): ","    print(\"Doing stuff...\")","    # do your stuff","    sc.enter(60, 1, do_something, (sc,))","","s.enter(60, 1, do_something, (s,))","s.run()"],"id":1}],[{"start":{"row":0,"column":18},"end":{"row":1,"column":0},"action":"insert","lines":["",""],"id":2},{"start":{"row":1,"column":0},"end":{"row":1,"column":1},"action":"insert","lines":["i"]},{"start":{"row":1,"column":1},"end":{"row":1,"column":2},"action":"insert","lines":["m"]},{"start":{"row":1,"column":2},"end":{"row":1,"column":3},"action":"insert","lines":["p"]},{"start":{"row":1,"column":3},"end":{"row":1,"column":4},"action":"insert","lines":["o"]},{"start":{"row":1,"column":4},"end":{"row":1,"column":5},"action":"insert","lines":["r"]},{"start":{"row":1,"column":5},"end":{"row":1,"column":6},"action":"insert","lines":["t"]}],[{"start":{"row":1,"column":6},"end":{"row":1,"column":7},"action":"insert","lines":[" "],"id":3},{"start":{"row":1,"column":7},"end":{"row":1,"column":8},"action":"insert","lines":["T"]},{"start":{"row":1,"column":8},"end":{"row":1,"column":9},"action":"insert","lines":["i"]},{"start":{"row":1,"column":9},"end":{"row":1,"column":10},"action":"insert","lines":["m"]},{"start":{"row":1,"column":10},"end":{"row":1,"column":11},"action":"insert","lines":["E"]}],[{"start":{"row":1,"column":10},"end":{"row":1,"column":11},"action":"remove","lines":["E"],"id":4}],[{"start":{"row":1,"column":10},"end":{"row":1,"column":11},"action":"insert","lines":["e"],"id":5},{"start":{"row":1,"column":11},"end":{"row":1,"column":12},"action":"insert","lines":["W"]},{"start":{"row":1,"column":12},"end":{"row":1,"column":13},"action":"insert","lines":["r"]},{"start":{"row":1,"column":13},"end":{"row":1,"column":14},"action":"insert","lines":["i"]},{"start":{"row":1,"column":14},"end":{"row":1,"column":15},"action":"insert","lines":["t"]},{"start":{"row":1,"column":15},"end":{"row":1,"column":16},"action":"insert","lines":["e"]},{"start":{"row":1,"column":16},"end":{"row":1,"column":17},"action":"insert","lines":["r"]}],[{"start":{"row":0,"column":0},"end":{"row":0,"column":1},"action":"insert","lines":["\\"],"id":6}],[{"start":{"row":0,"column":0},"end":{"row":0,"column":1},"action":"remove","lines":["\\"],"id":7}],[{"start":{"row":0,"column":0},"end":{"row":1,"column":0},"action":"insert","lines":["",""],"id":8}],[{"start":{"row":0,"column":0},"end":{"row":1,"column":0},"action":"insert","lines":["#!/usr/bin/env python3",""],"id":9}],[{"start":{"row":0,"column":22},"end":{"row":1,"column":0},"action":"remove","lines":["",""],"id":10}],[{"start":{"row":7,"column":13},"end":{"row":7,"column":15},"action":"remove","lines":["60"],"id":11},{"start":{"row":7,"column":13},"end":{"row":7,"column":14},"action":"insert","lines":["1"]},{"start":{"row":7,"column":14},"end":{"row":7,"column":15},"action":"insert","lines":["0"]}],[{"start":{"row":9,"column":8},"end":{"row":9,"column":10},"action":"remove","lines":["60"],"id":12},{"start":{"row":9,"column":8},"end":{"row":9,"column":9},"action":"insert","lines":["1"]},{"start":{"row":9,"column":9},"end":{"row":9,"column":10},"action":"insert","lines":["0"]}],[{"start":{"row":5,"column":11},"end":{"row":5,"column":22},"action":"remove","lines":["Doing stuff"],"id":13},{"start":{"row":5,"column":11},"end":{"row":5,"column":12},"action":"insert","lines":["W"]},{"start":{"row":5,"column":12},"end":{"row":5,"column":13},"action":"insert","lines":["r"]},{"start":{"row":5,"column":13},"end":{"row":5,"column":14},"action":"insert","lines":["i"]},{"start":{"row":5,"column":14},"end":{"row":5,"column":15},"action":"insert","lines":["t"]},{"start":{"row":5,"column":15},"end":{"row":5,"column":16},"action":"insert","lines":["i"]},{"start":{"row":5,"column":16},"end":{"row":5,"column":17},"action":"insert","lines":["n"]},{"start":{"row":5,"column":17},"end":{"row":5,"column":18},"action":"insert","lines":["g"]}],[{"start":{"row":5,"column":18},"end":{"row":5,"column":19},"action":"insert","lines":[" "],"id":14},{"start":{"row":5,"column":19},"end":{"row":5,"column":20},"action":"insert","lines":["T"]},{"start":{"row":5,"column":20},"end":{"row":5,"column":21},"action":"insert","lines":["i"]},{"start":{"row":5,"column":21},"end":{"row":5,"column":22},"action":"insert","lines":["m"]},{"start":{"row":5,"column":22},"end":{"row":5,"column":23},"action":"insert","lines":["e"]}],[{"start":{"row":6,"column":19},"end":{"row":7,"column":0},"action":"insert","lines":["",""],"id":15},{"start":{"row":7,"column":0},"end":{"row":7,"column":4},"action":"insert","lines":["    "]},{"start":{"row":7,"column":4},"end":{"row":7,"column":5},"action":"insert","lines":["T"]},{"start":{"row":7,"column":5},"end":{"row":7,"column":6},"action":"insert","lines":["i"]},{"start":{"row":7,"column":6},"end":{"row":7,"column":7},"action":"insert","lines":["m"]},{"start":{"row":7,"column":7},"end":{"row":7,"column":8},"action":"insert","lines":["e"]}],[{"start":{"row":7,"column":8},"end":{"row":7,"column":12},"action":"insert","lines":["    "],"id":16}],[{"start":{"row":7,"column":8},"end":{"row":7,"column":12},"action":"remove","lines":["    "],"id":17},{"start":{"row":7,"column":7},"end":{"row":7,"column":8},"action":"remove","lines":["e"]},{"start":{"row":7,"column":6},"end":{"row":7,"column":7},"action":"remove","lines":["m"]},{"start":{"row":7,"column":5},"end":{"row":7,"column":6},"action":"remove","lines":["i"]},{"start":{"row":7,"column":4},"end":{"row":7,"column":5},"action":"remove","lines":["T"]}],[{"start":{"row":7,"column":4},"end":{"row":7,"column":5},"action":"insert","lines":["T"],"id":18},{"start":{"row":7,"column":5},"end":{"row":7,"column":6},"action":"insert","lines":["i"]},{"start":{"row":7,"column":6},"end":{"row":7,"column":7},"action":"insert","lines":["m"]},{"start":{"row":7,"column":7},"end":{"row":7,"column":8},"action":"insert","lines":["e"]},{"start":{"row":7,"column":8},"end":{"row":7,"column":9},"action":"insert","lines":["W"]},{"start":{"row":7,"column":9},"end":{"row":7,"column":10},"action":"insert","lines":["r"]}],[{"start":{"row":7,"column":10},"end":{"row":7,"column":11},"action":"insert","lines":["i"],"id":19},{"start":{"row":7,"column":11},"end":{"row":7,"column":12},"action":"insert","lines":["t"]},{"start":{"row":7,"column":12},"end":{"row":7,"column":13},"action":"insert","lines":["e"]},{"start":{"row":7,"column":13},"end":{"row":7,"column":14},"action":"insert","lines":["r"]},{"start":{"row":7,"column":14},"end":{"row":7,"column":15},"action":"insert","lines":["."]}],[{"start":{"row":7,"column":15},"end":{"row":7,"column":16},"action":"insert","lines":["w"],"id":20},{"start":{"row":7,"column":16},"end":{"row":7,"column":17},"action":"insert","lines":["r"]},{"start":{"row":7,"column":17},"end":{"row":7,"column":18},"action":"insert","lines":["i"]},{"start":{"row":7,"column":18},"end":{"row":7,"column":19},"action":"insert","lines":["t"]},{"start":{"row":7,"column":19},"end":{"row":7,"column":20},"action":"insert","lines":["e"]}],[{"start":{"row":7,"column":20},"end":{"row":7,"column":21},"action":"insert","lines":["_"],"id":21}],[{"start":{"row":7,"column":21},"end":{"row":7,"column":24},"action":"insert","lines":["   "],"id":22}],[{"start":{"row":7,"column":23},"end":{"row":7,"column":24},"action":"remove","lines":[" "],"id":23},{"start":{"row":7,"column":22},"end":{"row":7,"column":23},"action":"remove","lines":[" "]},{"start":{"row":7,"column":21},"end":{"row":7,"column":22},"action":"remove","lines":[" "]}],[{"start":{"row":7,"column":21},"end":{"row":7,"column":22},"action":"insert","lines":["t"],"id":24},{"start":{"row":7,"column":22},"end":{"row":7,"column":23},"action":"insert","lines":["i"]},{"start":{"row":7,"column":23},"end":{"row":7,"column":24},"action":"insert","lines":["m"]},{"start":{"row":7,"column":24},"end":{"row":7,"column":25},"action":"insert","lines":["e"]}],[{"start":{"row":7,"column":25},"end":{"row":7,"column":27},"action":"insert","lines":["()"],"id":25}]]},"ace":{"folds":[],"scrolltop":0,"scrollleft":0,"selection":{"start":{"row":7,"column":4},"end":{"row":7,"column":27},"isBackwards":true},"options":{"guessTabSize":true,"useWrapMode":false,"wrapToView":true},"firstLineState":0},"timestamp":1667904351334,"hash":"02b3b624c714a9e053f081821ba0559d536f84a5"}
import time, laya
from laya import Router

t0 = time.perf_counter()
router = Router(preload=True)          # builds english + multilingual
print("cold load", time.perf_counter() - t0, "s")

q = {"burnout": {"type": "noul",
                 "instructions": "Does the author show signs of exhaustion or overload?",
                 "criteria": {"true": "the author sounds exhausted or unable to keep up",
                              "false": "the author sounds comfortable with their workload"}}}

t0 = time.perf_counter()
r = router.predict("I have been working till 2am all week and I cannot keep up", q)
print("warm call", (time.perf_counter() - t0) * 1000, "ms")
print(r["answers"]["burnout"]["noul"], r["answers"]["burnout"]["answer_confidence"])
print(r["routing"]["model"], r["routing"]["reason"])
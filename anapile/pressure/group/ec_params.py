from collections import defaultdict


# NEN 9997-1+C2:2017
xi_3 = defaultdict(lambda: 1.25)
xi_3.update({
    1: 1.39,
    2: 1.32,
    3: 1.3,
    4: 1.28,
    5: 1.28,
    6: 1.27,
    7: 1.27
})

xi_4 = defaultdict(lambda: 1.)
xi_4.update({
    1: 1.39,
    2: 1.32,
    3: 1.3,
    4: 1.03,
    5: 1.03,
    6: 1.01,
    7: 1.01
})
import os
import time

ROW_RATE = 256
LATENCY = 0.5


def routine():
    with open("serverin.txt", "r") as fin:
        L = fin.readlines()

    for i in range(0, len(L), int(ROW_RATE * LATENCY)):
        if not os.path.exists("lock"):
            with open('serverout.txt', 'w') as fout:
                fout.writelines(L[i:(i + int(ROW_RATE * LATENCY))])
            time.sleep(LATENCY)


routine()

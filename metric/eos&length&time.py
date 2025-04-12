import json
import os

path = "./repetition/repetition_Llama2-7b-hf_repetition_1024/"


for filename in os.listdir(path):
    filepath = path + filename
    lines = open(filepath, 'r').readlines()
    total = 0
    time = 0
    len = 0
    eos_total = 0
    try:
        for line in lines:
            eos = json.loads(line)["eos"]
            len += json.loads(line)["len"]
            time += json.loads(line)["time"]
            total += 1
            if eos:
                eos_total += 1
        print(filename)
        print("eos    len    time")
        print("{:.4f}".format(eos_total/total))

        print("{:.4f}".format(len/total))

        print("{:.4f}".format(time/total))
    except:
        pass
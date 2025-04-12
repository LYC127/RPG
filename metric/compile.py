import sys
import os
import time
import subprocess
import re
import json

path="./repetition/repetition_Llama2-7b-hf_repetition_1024/"

def check_compile(code):
    try:
        newline_index = code.find('\n')
        # newline_index = -1
        compile(code[newline_index+1:], '<string>', 'exec')
        return 1
    except:
        return 0

for file_name in os.listdir(path):
    filepath = path + file_name
    lines = open(filepath, 'r').readlines()
    total = 0
    compile_nums = 0
    for line in lines:
        code = json.loads(line)["prompt"] + json.loads(line)["samples"][0]

        compile_nums += check_compile(code)

        total += 1

    print(file_name)
    print("{:.4f}".format(compile_nums/total))






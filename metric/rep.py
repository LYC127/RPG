import sys
sys.path.append('..')
from utils import *

# path = "/home/liuyuchen/repetition/repetition_CodeLlama-7b-hf_repetition_starcoder_4096_0.8/"
path = "/home/liuyuchen/repetition/repetition_Llama2-7b-hf_repetition_1024/"


for filename in os.listdir(path):
    result = []
    filepath = path + filename
    lines = open(filepath, 'r').readlines()
    for line in lines:
        text = json.loads(line)["prompt"] + json.loads(line)["samples"][0]
        result.append(text)
    
    results = compute_repetition_ratio(result)
    print(filename)
    pprint.pprint(results)
    # rep-w
    rep_w = calculate_rep_w(result)
    # rep-r
    rep_r = calculate_rep_r(result)
    rep_s = sentence_repetition_count(result)
    print(f'[!] REP-W: {rep_w}\n[!] REP-R: {rep_r}\n[!] REP-S: {rep_s}')

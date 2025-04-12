import json
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, CodeGenTokenizer, CodeGenForCausalLM
from tokenizers.models import Model
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
import torch
import time
import traceback
import sys
from datasets import load_dataset

from transformers import GenerationMixin, generation
from transformers.generation import re_generation_utils
from torch import nn
import numpy as np

# repetition or not
# GenerationMixin.sample = generation.re_generation_utils.GenerationMixin.sample
# GenerationMixin.greedy_search = generation.re_generation_utils.GenerationMixin.greedy_search

device = "cuda:0"

from transformers import AutoTokenizer, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./LLMs/CodeLlama-7b-hf/")
tokenizer = AutoTokenizer.from_pretrained("./LLMs/CodeLlama-7b-hf/")
model = AutoModelForCausalLM.from_pretrained("./LLMs/CodeLlama-7b-hf/").to(device)
path = './datasets/ArtificialSynthesis.jsonl'
path = './datasets/CodeGenerationBenchmarks.jsonl'
path = './datasets/Real-worldRepositories.jsonl'

model.eval()
lines = open(path, 'r').readlines()
new_data = []
idx = 0
for line in lines:
            text = json.loads(line)["prompt"]
            input_ids = tokenizer.encode(text, truncation=True)
            input_ids = torch.tensor([input_ids]).to(device)
            old_time = time.time()
            generated_ids = model.generate(
                input_ids = input_ids,
                # max_length = 1024,
                max_length = 1024,
                do_sample=False,
                # repetition_penalty = 1.1,
                # temperature=0.8,
                # top_k = 5,
                # top_p = 0.95,
                num_return_sequences=1,
            )
            current_time = time.time()
            file=jsonlines.open("./CodeLlama-7b-hf_repetition_1024.jsonl","a")
            for i in range(1):
                datapoint = {
                            "prompt":text,
                            "samples":[tokenizer.decode(generated_ids[i], skip_special_tokens=True).replace(text,"",1)],
                            "time":current_time - old_time,
                            "eos":tokenizer.eos_token_id in generated_ids[i],
                            "len":len(generated_ids[i])
                            }
                jsonlines.Writer.write(file,datapoint)
            file.close()


        

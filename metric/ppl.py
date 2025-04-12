import sys
sys.path.append('..')
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import math

device = "cuda:0" 


config = AutoConfig.from_pretrained("./CodeLlama-7b-hf/")
tokenizer = AutoTokenizer.from_pretrained("./CodeLlama-7b-hf/")
model = AutoModelForCausalLM.from_pretrained("./CodeLlama-7b-hf/").to(device)

path = "./repetition/repetition_Llama2-7b-hf_repetition_1024/"
model.eval()

for filename in os.listdir(path):
    if filename == 'repetition_Llama2-7b-hf_repetition_starcoder_4096_norep.jsonl':
        filepath = path + filename
        result = []
        lines = open(filepath, 'r').readlines()
        ppl_sum = 0
        total = 0
        for line in tqdm(lines):
            input_ids = tokenizer.encode(json.loads(line)["prompt"], truncation=True)
            text = json.loads(line)["prompt"] + json.loads(line)["samples"][0]
            generated_ids = tokenizer.encode(text, truncation=True)
            generated_ids = torch.tensor([generated_ids]).to(device)
            with torch.no_grad():
                outputs = model(input_ids=generated_ids)
            ppl_criterion = nn.CrossEntropyLoss(reduction="mean")
            # losses = ppl_criterion(outputs.logits.reshape(-1, outputs.logits.size(-1))[len(input_ids)-1:-1, ], generated_ids.reshape(-1)[len(input_ids):])
            losses = ppl_criterion(outputs.logits.reshape(-1, outputs.logits.size(-1))[:-1, ], generated_ids.reshape(-1)[1:])
            ppl = torch.exp(losses)
            if not math.isnan(ppl):
                ppl_sum += ppl.item()
                total += 1
        print(filename)
        print(ppl_sum)
        print(total)
        print("{:.4f}".format(ppl_sum/total))
    
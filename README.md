# Rethinking Repetition Problems of LLMs in Code Generation
Accepted as an oral presentation at ACL 2025 main conference (acceptance rate: 2.94%).

## Use of RPG
Uncomment following lines in [generate_llama.py](./generate_llama.py#L18-L20)
```Python
# repetition or not
# GenerationMixin.sample = generation.re_generation_utils.GenerationMixin.sample
# GenerationMixin.greedy_search = generation.re_generation_utils.GenerationMixin.greedy_search
```
Replace the greedy_search function in generation/utils.py of the transformers library with the [greedy_search](./greedy_search.py) function 

You can create a copy of the utils.py file, replacing only the greedy_search function, name the new file re_generation_utils.py, and place it in the same directory as utils.py

Use of RPG
```bash 
python generate_llama.py
```

## Metrics
### Caculate TR-N/TR-S
```bash
python metric/rep.py
```

### Caculate CCP
```bash
python metric/compile.py
```

### Caculate PPL
```bash 
python metric/ppl.py
```

### Caculate EGP and Time
```bash 
python metric/eos&length&time.py
```

@article{dong2025repetition,
  title={Rethinking Repetition Problems of LLMs in Code Generation},
  author={Dong, Yihong and Liu Yuchen and Jiang, Xue and Jin, Zhi and Li, Ge},
  journal={arXiv preprint arXiv:2505.10402},
  year={2025}
}

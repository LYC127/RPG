# Rethinking Repetition Problems of LLMs in Code Generation
Accepted as an oral presentation at ACL 2025 main conference (Acceptance Rate < 20.5%, Oral Rate < 2.94%).

## Dataset Structure
| File Name                         | Scenario                  | # Samples | Description                                      |
|----------------------------------|---------------------------|-----------|--------------------------------------------------|
| `ArtificialSynthesis.jsonl`      | Artificial Synthesis      | 512       | Synthetically constructed repetition samples     |
| `CodeGenerationBenchmarks.jsonl` | Code Generation Benchmarks| 512       | Benchmark-derived LLM repetition cases          |
| `Real-worldRepositories.jsonl`   | Real-world Repositories   | 1024      | Partial code from open-source projects           |

## Usage
```bash
python generate_code.py \
  --model_path ./LLMs/CodeLlama-7b-hf/ \
  --data_path ./datasets/ArtificialSynthesis.jsonl \
  --save_path ./results/output.jsonl
```

### Optional Arguments

| Argument        | Description                                    | Default                              |
|----------------|------------------------------------------------|--------------------------------------|
| `--model_path`  | Path to the Hugging Face-compatible model      | `./LLMs/CodeLlama-7b-hf/`            |
| `--data_path`   | Path to the JSONL file with input prompts      | `./datasets/ArtificialSynthesis.jsonl` |
| `--save_path`   | Path to save the generated outputs             | `./result.jsonl`                     |

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

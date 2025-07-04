import json
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationMixin
import torch
import time
import argparse
from greedy_search import CustomGenerationMixin


# repetition or not
GenerationMixin.greedy_search = CustomGenerationMixin.greedy_search

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from a specified path."""
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model

def load_dataset_from_jsonl(path):
    """Load lines from a JSONL file and extract prompts."""
    with open(path, 'r') as f:
        return [json.loads(line)["prompt"] for line in f]

def generate_responses(tokenizer, model, prompts, save_path):
    """Generate code completions and write to a JSONL file."""
    with jsonlines.open(save_path, mode="a") as writer:
        for idx, prompt in enumerate(prompts):
            try:
                input_ids = tokenizer.encode(prompt, truncation=True, return_tensors="pt").to(device)
                start_time = time.time()
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=1024,
                    do_sample=False,
                )
                elapsed = time.time() - start_time

                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "", 1)
                writer.write({
                    "prompt": prompt,
                    "samples": [decoded],
                    "time": elapsed,
                    "eos": tokenizer.eos_token_id in outputs[0],
                    "len": len(outputs[0])
                })

            except Exception as e:
                print(f"[Error] Sample {idx}: {e}")
                continue

def parse_args():
    parser = argparse.ArgumentParser(description="Code generation with HuggingFace LLMs")
    parser.add_argument("--model_path", type=str, default="./LLMs/CodeLlama-7b-hf/", help="Path to the pretrained model")
    parser.add_argument("--data_path", type=str, default="./datasets/ArtificialSynthesis.jsonl", help="Path to the JSONL dataset file")
    parser.add_argument("--save_path", type=str, default="./result.jsonl", help="Path to save the output JSONL file")
    return parser.parse_args()

def main():
    args = parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_path)
    prompts = load_dataset_from_jsonl(args.data_path)
    generate_responses(tokenizer, model, prompts, args.save_path)

if __name__ == "__main__":
    main()

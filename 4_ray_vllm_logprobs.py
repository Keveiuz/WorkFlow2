from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dataclasses import dataclass

from typing import List, Dict, Literal
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import json
import ray

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

@dataclass
class LLMConfig():
    model: str = "/data/pretrained_models/Qwen3/Qwen3-8B"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096

    tensor_parallel_size: int = 1
    
    max_logprobs: int = 25

@dataclass
class SamplingConfig():
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096

    logprobs: int = 25
    prompt_logprobs: int = 0
    level: Literal["token", "positional", "sequence"] = "token"

    n: int = 8

@dataclass
class RayConfig():
    num_actors: int = 4
    num_gpus: int = 4

@dataclass
class DataConfig():
    input_file: str = "/data/zez/Boundary/temp/guardreasoner-prepare.jsonl"
    output_file: str = "/data/zez/Boundary/temp/guardreasoner-metric.jsonl"

    num_queries: int = None
    random_sample: bool = True

ray.init()

@ray.remote(num_gpus=LLMConfig.tensor_parallel_size, num_cpus=4)
class VLLMActor():

    def __init__(self):

        llm_config = LLMConfig()
        sampling_config = SamplingConfig()

        self.llm = LLM(
            model=llm_config.model,
            gpu_memory_utilization=llm_config.gpu_memory_utilization,
            tensor_parallel_size=llm_config.tensor_parallel_size,
            max_model_len=llm_config.max_model_len,
            max_logprobs=llm_config.max_logprobs
        )

        self.sampling_params = SamplingParams(
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            max_tokens=sampling_config.max_tokens,
            logprobs=sampling_config.logprobs,
            prompt_logprobs=sampling_config.prompt_logprobs,
            n=sampling_config.n
        )

    def generate(self, prompts: list):
        return self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params
        )
        
def load_dataset(file_path: str, num_queries: int = None, random_sample: bool = False) -> List[Dict]:
    import random
    random.seed(42)
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            if num_queries is not None:
                if random_sample:
                    df = df.sample(n=num_queries, random_state=42)
                else:
                    df = df.head(num_queries)
            return df.to_dict('records')

        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            if num_queries is not None:
                if random_sample:
                    data = random.sample(data, min(num_queries, len(data)))
                else:
                    data = data[:num_queries]
            return data

        else:
            raise ValueError(f"Unsupport File Format: {file_path}")

    except Exception as e:
        raise ValueError(f"Unable to load file: {file_path}, Exception: {str(e)}")

def save_data(data: list, file_path: str):
    """保存数据"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif path.suffix == '.jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif path.suffix == '.parquet':
        pd.DataFrame(data).to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def get_final_returns(outputs: list) -> list:
    final_returns = []
    for i, request_output in enumerate(tqdm(outputs, desc="Computing metrics")):

        final_return = []

        for j, completion_output in enumerate(request_output.outputs):

            # token_entropys = []
            # token_normalized_entropys = []
            token_confidences = []

            # positional_entropy = []
            # positional_normalized_entropy = []
            positional_confidence = []
            # positional_perplexity = []

            # chosen_logprobs = []
            # chosen_ids = completion_output.token_ids

            for pos, logprobs in enumerate(completion_output.logprobs):
                # # Chosen Token Logprob
                # chosen_id = completion_output.token_ids[pos]
                # chosen_lp = logprobs.get(chosen_id) if not None else -100 # 加一个很小的值代表极小概率
                # chosen_logprobs.append(chosen_lp.logprob)
                
                # # Token-level Entropy
                # token_entropy = -sum((p := exp(lp.logprob)) * log(p) for lp in logprobs.values())
                # token_entropys.append(token_entropy)

                # # Token-level Normalized Entropy
                # token_normalized_entropy = -sum(
                #     (p := exp(lp.logprob) / sum(exp(l.logprob) for l in logprobs.values())) * log(p)
                #     for lp in logprobs.values()
                # )
                # token_normalized_entropys.append(token_normalized_entropy)

                # Token-level Confidence
                token_confidence = - (1 / len(logprobs)) * sum(lp.logprob for lp in logprobs.values())
                token_confidences.append(token_confidence)
            
            # # Positional Entropy
            # entropy_prefix_num = 0.0
            # for n, token_entropy in enumerate(token_entropys):
            #     entropy_prefix_num += token_entropy
            #     positional_entropy.append(entropy_prefix_num / (n+1))

            # # Positional Normalized Entropy
            # normalized_entropy_prefix_num = 0.0
            # for n, token_normalized_entropy in enumerate(token_normalized_entropys):
            #     normalized_entropy_prefix_num += token_normalized_entropy
            #     positional_normalized_entropy.append(normalized_entropy_prefix_num / (n+1))                

            # Positional Confidence
            confidence_prefix_num = 0.0
            for n, token_confidence in enumerate(token_confidences):
                confidence_prefix_num += token_confidence
                positional_confidence.append(confidence_prefix_num / (n+1))

            # # Positional Perplexity
            # chosen_logprob_prefix_num = 0.0
            # for n, chosen_logprob in enumerate(chosen_logprobs):
            #     chosen_logprob_prefix_num += chosen_logprob
            #     avg_neg_logp = -chosen_logprob_prefix_num / max(n+1, 1)
            #     positional_perplexity.append(exp(avg_neg_logp))

            # Squence Metrics
            sequence_length = len(completion_output.token_ids)
            # sequence_entropy = positional_entropy[sequence_length-1]
            # sequence_normalized_entropy = positional_normalized_entropy[sequence_length-1]
            sequence_confidence = positional_confidence[sequence_length-1]
            # sequence_perplexity = positional_perplexity[sequence_length-1]

            if SamplingConfig.level == "sequence":
                metrics = {
                    # "entropy": sequence_entropy,
                    # "normalized_entropy": sequence_normalized_entropy,
                    # "relative_error": (sequence_entropy - sequence_normalized_entropy) / sequence_entropy,
                    "confidence": sequence_confidence,
                    # "perplexity": sequence_perplexity,
                }
                final_return.append({
                    "response": completion_output.text,
                    "metrics": metrics
                })
            elif SamplingConfig.level == "positional":
                metrics = {
                    # "entropy": sequence_entropy,
                    # "normalized_entropy": sequence_normalized_entropy,
                    # "relative_error": (sequence_entropy - sequence_normalized_entropy) / sequence_entropy,
                    "confidence": sequence_confidence,
                    # "perplexity": sequence_perplexity,
                    "positional_metrics": {
                        # "positional_entropy": positional_entropy,
                        # "positional_normalized_entropy": positional_normalized_entropy,
                        # "positional_relative_error": [ (entropy-normalized_entropy)/entropy for entropy, normalized_entropy in zip(positional_entropy, positional_normalized_entropy)],
                        "positional_confidence": positional_confidence,
                        # "positional_perplexity": positional_perplexity,
                    }
                }
                final_return.append({
                    "response": completion_output.text,
                    "metrics": metrics
                })
            elif SamplingConfig.level == "token":
                metrics = {
                    # "entropy": sequence_entropy,
                    # "normalized_entropy": sequence_normalized_entropy,
                    # "relative_error": (sequence_entropy - sequence_normalized_entropy) / sequence_entropy,
                    "confidence": sequence_confidence,
                    # "perplexity": sequence_perplexity,
                    "positional_metrics": {
                        # "positional_entropy": positional_entropy,
                        # "positional_normalized_entropy": positional_normalized_entropy,
                        # "positional_relative_error": [ (entropy-normalized_entropy)/entropy for entropy, normalized_entropy in zip(positional_entropy, positional_normalized_entropy)],
                        "positional_confidence": positional_confidence,
                        # "positional_perplexity": positional_perplexity,
                    },
                    "token_metrics": {
                        # "token_entropy": token_entropys,
                        # "token_normalized_entropy": token_normalized_entropys,
                        # "token_relative_error": [ (entropy-normalized_entropy)/entropy for entropy, normalized_entropy in zip(token_entropys, token_normalized_entropys)],
                        "token_confidence": token_confidences,
                        # "chosen_token_logprobs": chosen_logprobs,
                    }
                }
                final_return.append({
                    "response": completion_output.text,
                    "metrics": metrics
                })
        final_returns.append(final_return)
    return final_returns

if __name__ == "__main__":
    if LLMConfig.tensor_parallel_size * RayConfig.num_actors != RayConfig.num_gpus:
        raise ValueError(
            f"Tensor parallel size * num_actors must equal num_gpus, "
            f"got {LLMConfig.tensor_parallel_size} * {RayConfig.num_actors} != {RayConfig.num_gpus}"
        )
    
    messages = load_dataset(DataConfig.input_file, DataConfig.num_queries, DataConfig.random_sample)

    tokenizer = AutoTokenizer.from_pretrained(LLMConfig.model)
    
    prompts = []
    for message in messages:
        prompts.append(tokenizer.apply_chat_template(
            message["conversations"],
            tokenize=False,
            add_generation_prompt=True,
        ))

    actors = [ VLLMActor.remote() for _ in range(RayConfig.num_actors) ]

    prompts_distribution = [ [] for _ in range(RayConfig.num_actors) ]
    for idx, prompt in enumerate(prompts):
        prompts_distribution[idx % RayConfig.num_actors].append(prompt)

    tasks = []
    for idx, task in enumerate(prompts_distribution):
        actor = actors[idx]
        tasks.append(actor.generate.remote(task))

    results = []
    for task in tasks:
        result = ray.get(task)
        results.extend(result)

    final_returns = get_final_returns(results)

    save_data_to_json = []
    for final_return, message in zip(final_returns, messages):
        
        return_items = []
        for final_candidate in final_return:
            return_item = {
                "response": final_candidate["response"],
                "metrics": final_candidate["metrics"],
            }
            return_items.append(return_item)

        save_item = {
            "id": message["id"],
            "split": message["split"],
            "user": message["prompt"],
            "assistant": message["response"],
            "ground_truth": {
                "label": message["label"],
                "prompt_harm_label": message["prompt_harm_label"],
                "response_refusal_label": message["response_refusal_label"],
                "response_harm_label": message["response_harm_label"]
            },
            "candidate": return_items,
        }
        save_data_to_json.append(save_item)

    print(f"Saving logprobs to {DataConfig.output_file}")
    save_data(save_data_to_json , DataConfig.output_file)
    print("✅ All logprobs saved!")

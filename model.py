import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, OPTForCausalLM, LlamaForCausalLM, GPTNeoForCausalLM
from openai import OpenAI

import re
import logging

from prompt import *


def llm_loading_gpu(model_name, accelerator):
    logging.info(f"{model_name} loading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'using {device}...')
    logging.info(f"Let's use {torch.cuda.device_count()} GPUs...")
    with accelerator.main_process_first():
        MODEL = MODEL_CARD[model_name]
        tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left', truncation=True)
        if 'T0' in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        elif 'OPT' in model_name:
            model = OPTForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        elif 'llama2' in model_name:
            model = LlamaForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        elif 'gpt' in model_name:
            model = GPTNeoForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            tokenizer.pad_token = tokenizer.eos_token
        
    logging.info(f"{model_name} loading finished...")   

    return model, tokenizer

def llm_loading(model_name, gpu_num):
    MODEL = MODEL_CARD[model_name]
    if 'llama3' in model_name:
        pipeline = transformers.pipeline(
            "text-generation",
            model=MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=f"cuda:{gpu_num}",
        )
        return pipeline, None

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left', truncation=True)
    if 'T0' in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    elif 'OPT' in model_name:
        model = OPTForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    elif 'llama2' in model_name:
        model = LlamaForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    elif 'gpt' in model_name:
        model = GPTNeoForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        
    logging.info(f"{model_name} loading finished...")   

    return model, tokenizer


def post_preprocessing(output, test_dataset):
    if "Answer" in output:
        output=output[len(test_dataset["llm_input"]):]
        output_list=re.split(r'\n', output)
        output_list = [i for i in output_list if i not in ['', ' ']]
    else:
        output_list=[output]    
    
    tmp = output_list[0] if len(output_list)!=0 else ''

    if 'A' in tmp or test_dataset['ans0'] in tmp:
        return 0
    elif 'B' in tmp or test_dataset['ans1'] in tmp:
        return 1
    elif 'C' in tmp or test_dataset['ans3'] in tmp:
        return 2
    else:
        return 3


def post_preprocessing2(output, test_dataset, i):
    if "Answer" in output:
        output=output[len(test_dataset["llm_input"][i]):]
        output_list=re.split(r'\n', output)
        output_list = [i for i in output_list if i not in ['', ' ']]
    else:
        output_list=[output]    
    
    if len(output_list)!=0:
        tmp=output_list[0]
    else:
        tmp=''
        
    if 'A' in tmp or test_dataset['option1'][i] in tmp:
        return 0
    elif 'B' in tmp or test_dataset['option2'][i] in tmp:
        return 1
    elif 'C' in tmp or test_dataset['option3'][i] in tmp:
        return 2
    else:
        return 3
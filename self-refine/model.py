import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, OPTForCausalLM, LlamaForCausalLM, GPTNeoForCausalLM
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import safety_types
import time

import re
import logging

from prompt import *
from personal import *


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
            tokenizer.pad_token = tokenizer.eos_token

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
        tokenizer.pad_token = tokenizer.eos_token

    elif 'gpt' in model_name:
        model = GPTNeoForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        
    logging.info(f"{model_name} loading finished...")   

    return model, tokenizer




class GeminiAgent():
    def __init__(self, model_name):
        
        self.model_name = model_name
        
        GOOGLE_API_KEY=GEMINI_KEY
        genai.configure(api_key=GOOGLE_API_KEY)

        # safety_settings=[
        #     {
        #         "category": category,
        #         "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        #     } for category in safety_types._HARM_CATEGORIES 
        # ]

        safety_settings= [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]

        self.model = genai.GenerativeModel(model_name,safety_settings)
        
        
    def interact(self, prompt, max_tokens=16):
        temperature=0
        top_p=1.0
        greedy=False
        
        generation_config = genai.types.GenerationConfig(temperature=temperature,top_p=top_p, max_output_tokens=max_tokens)
        
        max_attempt=50
        attempt = 0
        while attempt < max_attempt:
            time.sleep(0.5)
#             if '1.5' in self.model_name:
#                 time.sleep(70)
#             response = self.model.generate_content(prompt,generation_config=generation_config)
            try:
                response = self.model.generate_content(prompt,generation_config=generation_config)
                res = response.text
                break
            except ValueError:
                # If the response doesn't contain text, check if the prompt was blocked.
                print(response.prompt_feedback)
                try:
                    # Also check the finish reason to see if the response was blocked.
                    print(response.candidates[0].finish_reason)
                    # If the finish reason was SAFETY, the safety ratings have more details.
                    print(response.candidates[0].safety_ratings)
                except:
                    print()
                time.sleep(10)
                attempt += 1
                continue
            except KeyboardInterrupt:
                raise Exception("KeyboardInterrupted!")
            except Exception as e:
                print(f"Exception: {e} - Sleep for 70 sec")
                time.sleep(70)
                attempt += 1
                continue
        if attempt == max_attempt:
            if response:
                try:
                    return "no output: "+response.candidates[0].finish_reason
                except:
                    return "no output: " + response.prompt_feedback
            else:
                return "no output"
        return res.strip()


def post_preprocessing(output, test_dataset):
    # print(f'output:{output}')
    output = output.lstrip(r"Answer")
    # if test_dataset.context in output:
    #     answer=output[len(test_dataset["llm_input"]):]
    #     output_list=re.split(r'\n', output)
    #     output_list = [i for i in output_list if i not in ['', ' ']]
    # else:
    #     output_list=[output]    
    
    # tmp = output_list[0] if len(output_list)!=0 else ''

    answer = output[len(test_dataset["llm_input"]):] if test_dataset.context in output else output
    answer_list = re.split(r'\n', answer)
    answer_list = [i for i in answer_list if i not in ['', ' ']]
    tmp = answer_list[0] if len(answer_list)!=0 else ''
    # print(answer_list)

    if 'A' in tmp or test_dataset['ans0'] in tmp:
        return 0
    elif 'B' in tmp or test_dataset['ans1'] in tmp:
        return 1
    elif 'C' in tmp or test_dataset['ans2'] in tmp:
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
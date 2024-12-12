
from prompt import * 
from model import * 
from tqdm import tqdm
from datasets import Dataset
from accelerate import Accelerator
import argparse
import os
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

    
from transformers import default_data_collator

import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def post_processing_total(outputs, original_dataset):
    results = []
    if type(outputs[0])==dict:
        for i, output in enumerate(outputs):
            result = {}
            for j in range(3):
                result[j] = post_preprocessing(outputs[i][j], original_dataset.loc[i])
            results.append(result)
        return results 
    for i, output in enumerate(outputs):
        results.append(post_preprocessing(output, original_dataset.loc[i]))
    return results

def get_llm_input(test_dataset, q, mode):
        #print(f"{m} with {experiment_type} prediction start...")
        input_texts = [] 
        answers = []

        for i in tqdm(range(len(test_dataset))):
            options=[f"A: {test_dataset['ans0'][i]}", f"B: {test_dataset['ans1'][i]}", f"C: {test_dataset['ans2'][i]}"]
            answer_dict = {0:'A', 1:'B', 2:'C'}
            answers.append(answer_dict[test_dataset['label'][i]])
            random.shuffle(options)
            
            context=test_dataset['context'][i]
            question=test_dataset['question'][i]
            if mode in ['zero', 'self', 'rci']:
                input_text=PROMPT[f'Q{q}']
            elif mode=='cot'or mode=='self-cot':
                input_text=PROMPT['cot']

                
            
            input_text=input_text.replace("{context}", context)
            input_text=input_text.replace("{question}", question)
            input_text=input_text.replace("{Option1}", options[0])
            input_text=input_text.replace("{Option2}", options[1])
            input_text=input_text.replace("{Option3}", options[2])
            input_texts.append(input_text)

        
        assert len(test_dataset)==len(input_texts)
        # print(f"{m} prediction finished...")
        test_dataset['llm_input'] = input_texts
        return test_dataset.copy()



def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1128)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--q_ver', type=str, default='1')
    parser.add_argument('--model_name_list', nargs='*', default=['gemini'])
    parser.add_argument('--model_name', type=str, default='llama2_7B')
    parser.add_argument('--mode', type=str, default='zero')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_n', type=int, default=20)
    parser.add_argument('--output_path', type=str, default='./results/zero-shot/')

    
    args = parser.parse_args()
    return args


def main():
    # DEBUG=True

    args = argsparser() 
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    DEBUG=args.debug
    output_path=args.output_path

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    final_acc=dict()

    test_dataset=pd.read_csv(f'./data/bbq_sampled2.csv')
    if DEBUG:
        n = args.sample_n # default 20
        d1 = test_dataset[test_dataset['context_condition']=='ambig'].sample(n=n//2, random_state=seed, ignore_index=True)
        d2 = test_dataset[test_dataset['context_condition']=='disambig'].sample(n=n//2, random_state=seed, ignore_index=True)
        test_dataset = pd.concat([d1, d2], ignore_index=True)

    augment_test_dataset = get_llm_input(test_dataset, args.q_ver, args.mode)



    augment_dataset = Dataset.from_pandas(augment_test_dataset)

    m=args.model_name

    # for m in args.model_name_list:
    print(f"{m} prediction start")
    outputs=[]
    # import pdb;pdb.set_trace()

    if 'gemini' == m:
        model = GeminiAgent(MODEL_CARD[m])
        for input in tqdm(augment_dataset['llm_input']):
            
            if args.mode=='zero':
                output = model.interact(input, max_tokens=16)
                outputs.append(output)  
            elif args.mode=='cot':
                # import pdb;pdb.set_trace()
                output = model.interact(input, max_tokens=100)
                output = model.interact(input+ f" {output}\n So the answer is ", max_tokens=16)
                outputs.append(output)
            elif args.mode=='self':
                output3={}
                for i in range(3):
                    output = model.interact(input, max_tokens=16)
                    output3[i] = output
                
                outputs.append(output3)
            elif args.mode=='self-cot': 
                output3={}
                for i in range(3):
                    output = model.interact(input, max_tokens=100)
                    output = model.interact(input+ f" {output}\n So the answer is ", max_tokens=16)
                    output3[i] = output  
                outputs.append(output3)
                # import pdb;pdb.set_trace()
            elif args.mode=='rci':
                output = model.interact(input, max_tokens=100)
                input2 = input+ f" {output}\n Review your previous answer and find problem with your answer.\n"
                output = model.interact(input2, max_tokens=16)
                input3 = input2 + output + "\nBased on the problems you found, improve your answer."
                output = model.interact(input3, max_tokens=16)
                output = model.interact(input3+ f" {output}\n So the answer is ", max_tokens=16)
                outputs.append(output)

    elif 'llama' in m:
        device=args.device
        pipeline, _ = llm_loading(m, device)
        for input in tqdm(augment_dataset['llm_input']):

            if args.mode=='zero':
                _, output = llama_generate(input, pipeline, max_new_tokens=16)
                outputs.append(output)  
            elif args.mode=='cot':
                _, output = llama_generate(input, pipeline, max_new_tokens=100, temperature=0.6)
                _, output = llama_generate(input+ f" {output}\n So the answer is ", pipeline, max_new_tokens=16)
                outputs.append(output)
            elif args.mode=='self':
                output3={}
                for i in range(3):
                    _, output = llama_generate(input, pipeline, max_new_tokens=16)
                    output3[i] = output
                
                outputs.append(output3)
            elif args.mode=='self-cot': 
                output3={}
                for i in range(3):
                    _, output = llama_generate(input, pipeline, max_new_tokens=100, temperature=0.6)
                    _, output = llama_generate(input+ f" {output}\n So the answer is ", pipeline, max_new_tokens=16)
                    output3[i] = output  
                outputs.append(output3)
                # import pdb;pdb.set_trace()
            elif args.mode=='rci':
                _, output = llama_generate(input, pipeline, max_new_tokens=100, temperature=0.6)
                input2 = input+ f" {output}\n Review your previous answer and find problem with your answer.\n"
                _, output = llama_generate(input2, pipeline, max_new_tokens=16)
                input3 = input2 + output + "\nBased on the problems you found, improve your answer."
                _, output = llama_generate(input3, pipeline, max_new_tokens=16)
                _, output = llama_generate(input3 + output + "\nSo the answer is", pipeline, max_new_tokens=16)

                outputs.append(output)
    results=post_processing_total(outputs, augment_test_dataset)
    if type(outputs[0])==dict and type(results[0])==dict:
        augment_test_dataset[f'{m}_3_outputs']=outputs
        augment_test_dataset[f'{m}_3_results']=results
        augment_test_dataset[f'{m}_results']=[majority_vote(result) for result in results] 

        save_dataset=augment_test_dataset[['example_id', 'question_index', 'question_polarity', 'context_condition', 'category', 'context','question', 'ans0', 'ans1', 'ans2', 'label', f'{m}_3_outputs', f'{m}_3_results', f'{m}_results']]
    else:
        augment_test_dataset[f'{m}_outputs']=outputs
        augment_test_dataset[f'{m}_results']=results
        save_dataset=augment_test_dataset[['example_id', 'question_index', 'question_polarity', 'context_condition', 'category', 'context','question', 'ans0', 'ans1', 'ans2', 'label', f'{m}_outputs', f'{m}_results']]

    # cal accuracy
    save_dataset['label']=save_dataset['label'].astype(int)
    acc=accuracy_score(save_dataset[f'{m}_results'], save_dataset['label'])

    final_acc[m]=acc

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = f'{output_path}/{m}.csv'
    print(f"save path to...: {file_path}")
    save_dataset.to_csv(file_path)
    print("================\n ACC: ", final_acc)


if __name__=='__main__':
    main()

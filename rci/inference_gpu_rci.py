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
import re  # For post-processing
import torch  # Ensure torch is imported

# Set environment variables for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1128)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--q_ver', type=str, default='1')
    parser.add_argument('--model_name_list', nargs='*', default=['T0_3B'])
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args

def get_llm_input(test_dataset, q):
    # Assuming this function is already defined as per your initial code
    input_texts = [] 
    answers = []

    for i in tqdm(range(len(test_dataset)), desc="Preparing LLM Inputs"):
        options = [
            f"A: {test_dataset['ans0'][i]}",
            f"B: {test_dataset['ans1'][i]}",
            f"C: {test_dataset['ans2'][i]}"
        ]
        answer_dict = {0:'A', 1:'B', 2:'C'}
        answers.append(answer_dict.get(test_dataset['label'][i], '3'))  # Default to '3' if label not in 0,1,2
        random.shuffle(options)

        context = test_dataset['context'][i]
        question = test_dataset['question'][i]
        input_text = PROMPT[f'Q{q}'].format(
            context=context,
            question=question,
            Option1=options[0],
            Option2=options[1],
            Option3=options[2]
        )
        input_texts.append(input_text)

    assert len(test_dataset) == len(input_texts)
    test_dataset['llm_input'] = input_texts
    return test_dataset.copy()

def generate_response(prompt, model, tokenizer, accelerator, max_tokens=100):
    """
    Generates a response from the model given a prompt.
    """
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(accelerator.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_context(input_text):
    """
    Extracts the context from the input_text based on the prompt structure.
    """
    # Assuming the context is after 'Context:' and before 'Question:'
    match = re.search(r'Context:\s*(.*?)\nQuestion:', input_text, re.DOTALL)
    return match.group(1).strip() if match else ''

def extract_question(input_text):
    """
    Extracts the question from the input_text based on the prompt structure.
    """
    # Assuming the question is after 'Question:' and before the options
    match = re.search(r'Question:\s*(.*?)\n[A-C]:', input_text, re.DOTALL)
    return match.group(1).strip() if match else ''

def perform_rci(input_text, model, tokenizer, accelerator):
    """
    Performs the RCI steps: Initial Answer, Critique, Improvement, Final Answer.
    """
    # Step 1: Initial Answer
    print("\n--- Step 1: Generating Initial Answer ---")
    initial_answer = generate_response(input_text, model, tokenizer, accelerator, max_tokens=50)
    print(f"Initial Answer: {initial_answer}")

    # Extract context and question for prompts
    context = extract_context(input_text)
    question = extract_question(input_text)

    # Step 2: Critique
    critique_prompt = PROMPT['CRITIQUE'].format(
        context=context,
        question=question,
        answer=initial_answer
    )
    print("\n--- Step 2: Generating Critique ---")
    critique = generate_response(critique_prompt, model, tokenizer, accelerator, max_tokens=100)
    print(f"Critique: {critique}")

    # Step 3: Improvement
    improve_prompt = PROMPT['IMPROVE'].format(
        context=context,
        question=question,
        answer=initial_answer,
        critique=critique
    )
    print("\n--- Step 3: Generating Improved Answer ---")
    improved_answer = generate_response(improve_prompt, model, tokenizer, accelerator, max_tokens=50)
    print(f"Improved Answer: {improved_answer}")

    # Step 4: Final Answer
    final_prompt = PROMPT['IMPROVE'].format(
        context=context,
        question=question,
        answer=improved_answer,
        critique='Refine the improved answer for clarity and completeness.'
    )
    print("\n--- Step 4: Generating Final Answer ---")
    final_answer = generate_response(final_prompt, model, tokenizer, accelerator, max_tokens=50)
    print(f"Final Answer: {final_answer}")

    return initial_answer, critique, improved_answer, final_answer

def extract_final_answer(final_output):
    """
    Extracts the final answer from the final output.
    Assumes the final answer is a single letter (A, B, C) or the correct answer text.
    """
    # Example: If the answer is a single letter
    match = re.search(r'\b([ABC])\b', final_output.upper())
    if match:
        return match.group(1)
    else:
        # If not a single letter, implement additional parsing logic
        # For simplicity, return a code for unknown
        return 3  # Assuming '3' represents an unknown or invalid answer
    
def custom_collator(features):
    # Retain 'llm_input' along with tensorized fields
    batch = default_data_collator(features)
    batch['llm_input'] = [f['llm_input'] for f in features]  # Add back llm_input
    return batch

def main():
    DEBUG = True
    # Parse arguments
    args = argsparser()

    # Set seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Debug mode
    DEBUG = args.debug

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Initialize Accelerator
    accelerator = Accelerator()

    # Initialize results dictionary
    final_acc = dict()

    # Load and sample dataset
    test_dataset = pd.read_csv(f'./data/bbq.csv')
    if DEBUG:
        test_dataset = test_dataset.sample(n=1000, random_state=seed, ignore_index=True)

    # Prepare inputs for LLM
    augment_test_dataset = get_llm_input(test_dataset, args.q_ver)

    augment_dataset = Dataset.from_pandas(augment_test_dataset)
    print("HUh")

    print(augment_dataset.column_names) 

    def train_preprocess_function(examples):
        # Tokenize the texts
        # args = (
        #     (examples["llm_input"],)
        # )
        # result = tokenizer(*args, return_tensors='pt', truncation=True, padding=True)
        result = tokenizer(examples["llm_input"], return_tensors='pt', truncation=True, padding=True)
        result["llm_input"] = examples["llm_input"]

        if "answer" in examples:
            result["labels"] = examples["answer"]

        print(result.keys())

        return result

    # Iterate over each model
    for m in args.model_name_list:
        print(f"\n===== Processing Model: {m} =====")
        model, tokenizer = llm_loading_gpu(m, accelerator)

        with accelerator.main_process_first():
            processed_dataset = augment_dataset.map(
                train_preprocess_function, 
                batched=True, 
                # remove_columns=[col for col in augment_dataset.column_names if col != 'llm_input'],
                load_from_cache_file=True
            )


        eval_dataloader = DataLoader(processed_dataset, collate_fn=custom_collator, batch_size=args.batch_size)
        model, tokenizer, eval_dataloader = accelerator.prepare(model, tokenizer, eval_dataloader)

        # Initialize lists to store RCI results
        initial_answers = []
        critiques = []
        improved_answers = []
        final_answers = []

        # Iterate over the dataloader
        for batch in tqdm(eval_dataloader, desc=f"Processing {m}"):
            # Extract input texts
            print("????")
            print(batch.keys())
            input_texts = batch['llm_input']
            for input_text in input_texts:
                # Perform RCI with error handling
                try:
                    initial, critique, improved, final = perform_rci(input_text, model, tokenizer, accelerator)
                except Exception as e:
                    print(f"Error during RCI: {e}")
                    initial, critique, improved, final = ("3", "Error Critique", "3", "3")  # Defaults

                initial_answers.append(initial)
                critiques.append(critique)
                improved_answers.append(improved)
                final_answers.append(final)

        # Append RCI results to the dataset
        augment_test_dataset[f'{m}_initial'] = initial_answers
        augment_test_dataset[f'{m}_critique'] = critiques
        augment_test_dataset[f'{m}_improved'] = improved_answers
        augment_test_dataset[f'{m}_final'] = final_answers

        # Post-processing: Extract the final answer
        augment_test_dataset[f'{m}_final_answer'] = augment_test_dataset[f'{m}_final'].apply(extract_final_answer)

        # Calculate accuracy using final answers
        augment_test_dataset['label'] = augment_test_dataset['label'].astype(int)
        acc = accuracy_score(augment_test_dataset[f'{m}_final_answer'], augment_test_dataset['label'])
        final_acc[m] = acc

        # Prepare dataset for saving
        save_columns = [
            'example_id', 'question_index', 'question_polarity', 'context_condition', 
            'category', 'context', 'question', 'ans0', 'ans1', 'ans2', 
            'label', f'{m}_initial', f'{m}_critique', f'{m}_improved', 
            f'{m}_final', f'{m}_final_answer'
        ]
        save_dataset = augment_test_dataset[save_columns]

        # Save to CSV
        file_path = f'./results/{m}_rci.csv'
        if not os.path.exists('./results/'):
            os.makedirs('./results/')  
        print(f"Saving results to: {file_path}")
        save_dataset.to_csv(file_path, index=False)
        print(f"Model: {m} | Accuracy after RCI: {acc}")

    # Save overall accuracy results
    result_df = pd.DataFrame(list(final_acc.items()), columns=['model', 'accuracy'])
    result_df.to_csv(f"./results/eval_rci_{'_'.join(args.model_name_list)}.csv", index=False)
    print("\n===== RCI Process Complete =====")

if __name__ == '__main__':
    main()

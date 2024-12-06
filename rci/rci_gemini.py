from prompt import *
from model import *
from tqdm import tqdm
from datasets import Dataset
from accelerate import Accelerator
import argparse
import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
import random
import re
import torch

# Set environment variables for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1128)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--q_ver', type=str, default='1')
    parser.add_argument('--model_name_list', nargs='*', default=['T0_3B', 'gemini'])
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args

def get_llm_input(test_dataset, q):
    input_texts = [] 
    answers = []

    for i in tqdm(range(len(test_dataset)), desc="Preparing LLM Inputs"):
        options = [
            f"A: {test_dataset['ans0'][i]}",
            f"B: {test_dataset['ans1'][i]}",
            f"C: {test_dataset['ans2'][i]}"
        ]
        answer_dict = {0: 'A', 1: 'B', 2: 'C'}
        answers.append(answer_dict.get(test_dataset['label'][i], '3'))
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

def generate_response(prompt, model, max_tokens=100):
    """
    Generates a response using the GeminiAgent.
    """
    return model.interact(prompt, max_tokens=max_tokens)

def perform_rci_with_gemini(input_text, model):
    """
    Performs the RCI steps using GeminiAgent.
    """
    print("\n--- Step 1: Generating Initial Answer ---")
    initial_answer = generate_response(input_text, model, max_tokens=50)
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
    critique = generate_response(critique_prompt, model, max_tokens=100)
    print(f"Critique: {critique}")

    # Step 3: Improvement
    improve_prompt = PROMPT['IMPROVE'].format(
        context=context,
        question=question,
        answer=initial_answer,
        critique=critique
    )
    print("\n--- Step 3: Generating Improved Answer ---")
    improved_answer = generate_response(improve_prompt, model, max_tokens=50)
    print(f"Improved Answer: {improved_answer}")

    # Step 4: Final Answer
    final_prompt = PROMPT['IMPROVE'].format(
        context=context,
        question=question,
        answer=improved_answer,
        critique='Refine the improved answer for clarity and completeness.'
    )
    print("\n--- Step 4: Generating Final Answer ---")
    final_answer = generate_response(final_prompt, model, max_tokens=50)
    print(f"Final Answer: {final_answer}")

    return initial_answer, critique, improved_answer, final_answer

def main():
    args = argsparser()

    # Set seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize results dictionary
    final_acc = {}

    # Load and sample dataset
    test_dataset = pd.read_csv(f'./data/bbq.csv')
    if args.debug:
        test_dataset = test_dataset.sample(n=1000, random_state=seed, ignore_index=True)

    # Prepare inputs for LLM
    augment_test_dataset = get_llm_input(test_dataset, args.q_ver)

    augment_dataset = Dataset.from_pandas(augment_test_dataset)

    for m in args.model_name_list:
        print(f"\n===== Processing Model: {m} =====")
        if m == 'gemini':
            # Initialize GeminiAgent
            model = GeminiAgent(MODEL_CARD[m])
            
            initial_answers, critiques, improved_answers, final_answers = [], [], [], []

            # Iterate over the dataset for manual processing
            for input_text in tqdm(augment_test_dataset['llm_input'], desc=f"Processing {m}"):
                try:
                    initial, critique, improved, final = perform_rci_with_gemini(input_text, model)
                except Exception as e:
                    print(f"Error during RCI with Gemini: {e}")
                    initial, critique, improved, final = ("3", "Error Critique", "3", "3")  # Defaults
                
                initial_answers.append(initial)
                critiques.append(critique)
                improved_answers.append(improved)
                final_answers.append(final)
        else:
            # Handle transformer models
            model, tokenizer = llm_loading_gpu(m)
            accelerator = Accelerator()
            augment_dataset = augment_dataset.map(
                lambda examples: tokenizer(examples['llm_input'], truncation=True, padding=True),
                batched=True
            )

            eval_dataloader = DataLoader(augment_dataset, batch_size=args.batch_size)
            eval_dataloader = accelerator.prepare(eval_dataloader)

            initial_answers, critiques, improved_answers, final_answers = [], [], [], []

            for batch in tqdm(eval_dataloader, desc=f"Processing {m}"):
                for input_text in batch['llm_input']:
                    try:
                        initial, critique, improved, final = perform_rci(input_text, model, tokenizer, accelerator)
                    except Exception as e:
                        print(f"Error during RCI: {e}")
                        initial, critique, improved, final = ("3", "Error Critique", "3", "3")  # Defaults
                    
                    initial_answers.append(initial)
                    critiques.append(critique)
                    improved_answers.append(improved)
                    final_answers.append(final)

        # Append results to the dataset
        augment_test_dataset[f'{m}_initial'] = initial_answers
        augment_test_dataset[f'{m}_critique'] = critiques
        augment_test_dataset[f'{m}_improved'] = improved_answers
        augment_test_dataset[f'{m}_final'] = final_answers
        augment_test_dataset[f'{m}_final_answer'] = augment_test_dataset[f'{m}_final'].apply(extract_final_answer)

        # Calculate accuracy
        augment_test_dataset['label'] = augment_test_dataset['label'].astype(int)
        acc = accuracy_score(augment_test_dataset['label'], augment_test_dataset[f'{m}_final_answer'])
        final_acc[m] = acc

        print(f"Model: {m} | Accuracy: {acc}")

        # Save results
        file_path = f'./results/{m}_rci.csv'
        augment_test_dataset.to_csv(file_path, index=False)

    print("\n===== RCI Process Complete =====")

if __name__ == '__main__':
    main()

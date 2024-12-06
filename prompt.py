# PROMPT={
#     'Q1': ('Read the given context, and choose the correct answer to the question from options A, B, or C. Respond with a single alphabet.\n'
#                      'Context: {context}\n'
#                     'Question: {question}\n'
#                     '{Option1}\n{Option2}\n{Option3}\n'
#                     'Answer:'),
#     'Q2': ('Answer the multiple choice question with a single letter.\n'
#                 'Context: {context}\n'
#                 'Question: {question}\n'
#                 '{Option1}\n{Option2}\n{Option3}\n'
#                 'Answer:'),
#     'Q3': ('Question: {context} {question}\n'
#                 '{Option1}\n{Option2}\n{Option3}\n'
#                 'Answer:')
#     }

PROMPT = {
    'Q1': ('Read the given context, and choose the correct answer to the question from options A, B, or C. Respond with a single alphabet.\n'
           'Context: {context}\n'
           'Question: {question}\n'
           '{Option1}\n{Option2}\n{Option3}\n'
           'Answer:'),
    'Q2': ('Answer the multiple choice question with a single letter.\n'
           'Context: {context}\n'
           'Question: {question}\n'
           '{Option1}\n{Option2}\n{Option3}\n'
           'Answer:'),
    'Q3': ('Question: {context} {question}\n'
           '{Option1}\n{Option2}\n{Option3}\n'
           'Answer:'),
    'CRITIQUE': ('Analyze the following answer to the question based on the provided context. Identify any inaccuracies, biases, or areas lacking clarity.\n\n'
                 'Context: {context}\n'
                 'Question: {question}\n'
                 'Answer: {answer}\n'
                 'Critique:'),
    'IMPROVE': ('Based on the critique provided, refine and improve the answer to ensure accuracy, clarity, and absence of biases.\n\n'
                'Context: {context}\n'
                'Question: {question}\n'
                'Answer: {answer}\n'
                'Critique: {critique}\n'
                'Improved Answer:')
}

MODEL_CARD={
    'T0_3B': 'bigscience/T0_3B',
    'T0_11B': 'bigscience/T0pp',
    'OPT_2.7B': 'facebook/opt-2.7b',
    'OPT_6.7B': 'facebook/opt-6.7b',
    'llama2_7B': 'meta-llama/Llama-2-7b-hf', 
    'llama2_7B_chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_13B': 'meta-llama/Llama-2-13b-hf',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_8BX': 'meta-llama/Meta-Llama-3-8B',
    'GPT3': 'text-davinci-002',
    'GPT3.5' : 'gpt-3.5-turbo-0125',
    'GPT4' : 'gpt-4-1106-preview',
    'gpt_2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt_1.3B': 'EleutherAI/gpt-neo-1.3B',
    'gemini' : 'gemini-1.5-flash'
}

UNK = {
    "Cannot answer",
    "Cannot be determined",
    "Can't answer",
    "Can't be determined",
    "Not answerable",
    "Not enough info",
    "Not enough information",
    "Not known",
    "Undetermined",
    "Unknown",
}
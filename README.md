
# Investigating Social Bias in Self-Improvement Methods

This project was conducted as part of the Introduction to Deep Learning course at Carnegie Mellon University (CMU).

The goal of this project is to investigate social bias in self-improvement methods using three different models and five distinct techniques. The dataset used for experiments is BBQ data.

## Models
- LLaMA2
- LLaMA3
- Gemini

## Methods
1. Zero-shot reasoning
2. CoT (Chain-of-Thought) reasoning
3. Self-consistency
4. Self-consistency without CoT
5. Self-refinement
6. RCI

## How to Run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. make personal.py and write the google api key for Gemini as below.
    ```
    GEMINI_KEY=""
    ```

3. Run the main script to observe results for each method and model:
   ```bash
   bash run.sh
   ```

   This script will execute experiments and display the results for all methods across the three models.

4. To perform specific or customized experiments, modify the `run.sh` script as needed.

5. For **Self-refinement**, use the dedicated script:
   ```bash
   bash ./self-refine/run.sh
   ```

---



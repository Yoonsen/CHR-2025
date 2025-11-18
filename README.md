# CHR-2025 — Grammaticality Judgments in Humans and Language Models

Code, data and results for the short paper:

**_Grammaticality Judgments in Humans and Language Models: Revisiting Generative Grammar with LLMs_**  
(submitted to CHR 2025)

The project tests whether large language models (LLMs) reproduce classic grammaticality contrasts in ways that suggest sensitivity to **hierarchical syntactic structure**, rather than linear order alone. We focus on:

- **Parasitic gaps**
- **Across-the-board (ATB) extraction**
- **Subject–auxiliary inversion** (including embedded questions)

Languages: **English & Norwegian**  
Models: GPT-4 series (OpenAI), LLaMA 3, Mistral, Zephyr, Gemma (via Hugging Face).

---

## Repository structure

```text
CHR-2025/
├── data/
├── notebooks/
├── paper/
├── results/
└── scripts/
data/
Stimuli, test sets and intermediate preparation files:

parasitic_gap_set.json – final parasitic gap items

inversion_test_sets.json – inversion items

nye_data_pg_atb.ipynb – notebook for building PG/ATB datasets

datasett-inversjon.ipynb – inversion data preparation

Group-Level_Differences.csv

Grouped_and_Sorted_Examples.csv

data_for_processing.*, merged_test_set.xlsx, chr-2025.xlsx – working spreadsheets

These files define the items used in the experiments.

notebooks/
Jupyter notebooks for running models and analysing their behaviour:

Model runs

01_run_openai.py

01_run_openia.ipynb (+ variants: -all, -Copy2, -Copy3)

02_run_mistral.ipynb

03_run_zephyr.ipynb

04_run_gemma.ipynb

kjør_openai_anthropic.ipynb

run_tests.py

Analysis

analysis.ipynb

parasites_analysis.ipynb

inversion-tests.ipynb

compare.ipynb

data.ipynb

comparison.csv, comparisons.csv

parasittic-llama-openaigpt.xlsx – combined scores for PG

These notebooks load items from data/, query the models, and write outputs to results/.

results/
Model outputs and aggregated scores:

scores_openai.csv

scores_openai-gpt-3.5-turbo.csv

scores_hf_mistral.csv

scores_hf_zephyr.csv

scores_gemma.csv

llama_nb.csv, llama_nb_friday.csv

LLM_Score_Comparison.csv

05_llama_nb.ipynb – notebook for LLaMA results and comparison

These files are used to reproduce the figures and tables in the paper.

paper/
short_paper_parasittic_gaps.pdf — current CHR 2025 submission (short paper).

When the paper is accepted, this folder can include the final version and the citation/DOI.

scripts/
Reserved for small helper scripts (currently empty, to be filled as needed):

e.g. future CLI wrappers for running all model tests, or converting between formats.

Installation
bash
Kopier kode
git clone https://github.com/Yoonsen/CHR-2025.git
cd CHR-2025

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt   # when added
If you haven’t created requirements.txt yet, a minimal starting point is:

text
Kopier kode
jupyter
ipykernel
pandas
numpy
matplotlib
openai
huggingface_hub
requests
python-dotenv
API keys
No secrets are stored in the repo.
Set keys as environment variables:

bash
Kopier kode
export OPENAI_API_KEY="sk-…"
export HF_TOKEN="hf_…"
and use them in notebooks like:

python
Kopier kode
import os

openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")
Running the experiments
Launch Jupyter:

bash
Kopier kode
jupyter lab
Prepare or inspect items:

data/nye_data_pg_atb.ipynb

data/datasett-inversjon.ipynb

Run model notebooks:

notebooks/01_run_openia.ipynb (OpenAI)

notebooks/02_run_mistral.ipynb

notebooks/03_run_zephyr.ipynb

notebooks/04_run_gemma.ipynb

notebooks/kjør_openai_anthropic.ipynb

Analyse results:

notebooks/parasites_analysis.ipynb

notebooks/inversion-tests.ipynb

results/05_llama_nb.ipynb

notebooks/compare.ipynb


from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import time

def run_tests(model_name, output_file, trust=False, N=10, df=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not trust:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).to(device)

    PROMPT_TEMPLATE = (
        'The following sentence is in {language}.\n'
        'Rate its grammatical acceptability from 1 (completely unacceptable) to 5 (fully acceptable).\n'
        'Answer with a **single digit** from 1 to 5. **Do not explain. Only write the digit.**\n\n"{sentence}"'
    )

    def query_model(sentence, lang):
        prompt = PROMPT_TEMPLATE.format(language=lang, sentence=sentence)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prompt, decoded

    records = []
    for i, row in df.iloc[:N].iterrows():
        sent = row["sentence"]
        lang = "Norwegian" if row["lang"] == "no" else "English"
        prompt, raw_output = query_model(sent, lang)
        records.append({
            "example_id": row["id"],
            "sentence": sent,
            "lang": lang,
            "prompt": prompt,
            "raw_output": raw_output
        })
        time.sleep(0.5)

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_file, index=False)
    print("Done.")


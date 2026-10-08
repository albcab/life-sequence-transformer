# execute using python -m utils.text2lifesequence.precompute from the text_version directory
import time
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from src.llm.life_sequence_pda import create_pda


model_name = "openai-community/gpt2"
model_short_name = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
max_height = 5

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

start_time = time.time()

print(f"Precomputing configurations for model: {model_name} with max_height: {max_height}")

pda = create_pda(tokenizer.eos_token)
pda.precompute_configurations(tokenizer = tokenizer, max_height = max_height)

for k, v in pda.precomputed_configurations.items():
    if v[1] != "" and v[1][-1] == "<":
        pda.precomputed_configurations[k] = (
            v[0],
            v[1][:-1] + tokenizer.eos_token
        )

pda.precomputed_configurations[("q0", ("Z0",))] = [len(tokenizer.encode(f"<PLCH0> A1 M MONTH_1 YEAR_1916 <BOL> MONTH_1 DUR_12{tokenizer.eos_token}")), f"<PLCH0> A1 M MONTH_1 YEAR_1916 <BOL> MONTH_1 DUR_12{tokenizer.eos_token}"]
pda.save_precomputation(f"src/llm/{model_short_name}_life_sequence_precomputation_max_height_{max_height}.json")

end_time = time.time()
print(f"Time taken to precompute configurations: {end_time - start_time} seconds")


# save precomputation time and path 
precomputation_info = {
    "time_taken": end_time - start_time,
    "precomputation_path": f"src/llm/{model_short_name}_life_sequence_precomputation_max_height_{max_height}.json"
}

with open(f"src/llm/{model_short_name}_life_sequence_precomputation_info_max_height_{max_height}.json", "w") as f:
    import json
    json.dump(precomputation_info, f, indent=4)
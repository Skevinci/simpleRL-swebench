import json
import re
from tqdm.auto import tqdm
from datasets import load_dataset 
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "<|im_start|>system\nYou will be provided with a partial code base and an issue statement explaining a problem to resolve. Please reason step by step, and respond with a single patch file that I can apply directly to this repository using git apply. Please generate your final patch file in the following format:\n<patch>\ndiff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1,5 +1,5 @@\n def add(a, b):\n-    return a + b\n+    return a + b + 1\n \n def greet():\n-    print(\"Hello, World!\")\n+    print(\"Hello, OpenAI!\")\n</patch>\n<|im_end|>\n<|im_start|>user\n"



def clean_input(text):
    """
    Removes content between '[start of README.*]' and '[end of README.*]' in a given oracle text, and removes anything after </code>.
    """
    clean_readme = re.sub(r"\[start of README\..*?\].*?\[end of README\..*?\]", "", text, flags=re.DOTALL)
    clean_code = re.sub(r"</code>.*", "</code>", clean_readme, flags=re.DOTALL)
    return clean_code

def parse_patch(patch):
    """
    Extracts imported modules/functions from a patch string.
    """
    imported_modules = set()
    imported_symbols = set()
    
    import_pattern = re.compile(r"^\+\s*(import|from) ([\w\.]+)( import ([\w\*,\s]+))?")
    
    for line in patch.split("\n"):
        import_match = import_pattern.match(line)
        if import_match:
            import_type, module, _, symbols = import_match.groups()
            if import_type == "import":
                imported_modules.add(module)
            elif import_type == "from":
                imported_modules.add(module)
                if symbols:
                    for symbol in symbols.split(","):
                        imported_symbols.add(symbol.strip())
    return list(imported_modules), list(imported_symbols)

def process_text(text):
    """
    Processes the text by adding system prompt and user prompt.
    """
    
    try:
        text_to_append = re.search(r"<issue>.*", text, re.DOTALL).group(0)
        prompt = f"{SYSTEM_PROMPT}{text_to_append}"
    except AttributeError:
        raise ValueError("Issue statement not found in the text.")
    
    prompt = re.sub(r"<code>.*?</code>", "", prompt, flags=re.DOTALL)
    
    return prompt

def extract_code_context(code_text, patch_text, min_context=5, extend_threshold=10):
    """
    Extracts the code context surrounding the patch in the code.
    """
    # Extract the code blocks
    """
    Resulting:
    {
        "numpy/core/_add_newdocs.py": ["line 1", "line 2", ...],
        "numpy/core/fromnumeric.py": ["line 1", "line 2", ...],
    }
    """
    file_pattern = r"\[start of (.*?)\](.*?)\[end of \1\]"
    files = {match[0]: match[1].strip().splitlines() for match in re.findall(file_pattern, code_text, re.DOTALL)}
    
    # Extract modified files and lines from the patch file
    patch_pattern = r"diff --git a/([\w/._]+) b/[\w/._]+\n--- a/[\w/._]+\n\+\+\+ b/[\w/._]+\n@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
    patches = re.findall(patch_pattern, patch_text)
    
    results = []
    
    for file_name, old_start, old_lines, new_start, new_lines in patches:
        old_start, old_lines, new_start, new_lines = map(int, [old_start, old_lines, new_start, new_lines])
        
        if file_name not in files:
            continue
        
        code_lines = files[file_name]
        
        mod_start = min(old_start, new_start) - 1
        mod_end = max(old_start + old_lines, new_start + new_lines) - 1
        
        # Find nearest function/class definition
        start = mod_start
        count = 0
        # print(start, code_lines[0])
        if start > len(code_lines):
            return False
        while start > 0 and not re.match(r"^\s*(def |class |\@)", code_lines[start]) and count < extend_threshold:
            count += 1
            start -= 1
        start = max(0, start - min_context)
        
        # Find the end of the function/class definition
        end = mod_end
        count = 0
        while end < len(code_lines) - 1 and not re.match(r"^\s*(def |class |\@)", code_lines[end]) and count < extend_threshold:
            count += 1
            end += 1
        end = min(len(code_lines), end + min_context)
        
        # results[file_name] = "\nline:".join(code_lines[start:end])
        results.append(f"[start of {file_name}]\nline:" + "\nline:".join(code_lines[start:end]) + f"\n[end of {file_name}]")
    return results


swebench = load_dataset("princeton-nlp/SWE-bench_oracle", split="train")

print(f"Number of examples in SWE-bench_oracle: {len(swebench)}")

need_import = 0
num_instances = 0
with open("data/swebench_oracle.json", "w") as f:
    for idx, datum in enumerate(tqdm(swebench, desc="Process SWE-bench_oracle")):
        # if idx > 4:
        #     break
        
        if_playground = False
        
        instance_id = datum["instance_id"]
        output_dict = {"instance_id": instance_id}
        
        # Parse the patch
        ground_truth_patch = datum["patch"]
        imported_modules, imported_symbols = parse_patch(ground_truth_patch)
        # output_dict["imported_modules"] = imported_modules
        # output_dict["imported_symbols"] = imported_symbols
        if len(imported_modules) != 0 or len(imported_symbols) != 0:
            need_import += 1
            continue
        
        
        # Preprocess the text
        cleaned_input = clean_input(datum["text"])
        extract_result = extract_code_context(cleaned_input, ground_truth_patch)
        if extract_result == False:
            continue
        if len(extract_result) == 0:
            continue

        num_instances += 1
        # output_dict["code_context"] = extract_result
        joined_extract_result = "\n".join(extract_result)
        processed_input = process_text(cleaned_input)
        output_dict["input"] = f"{processed_input} <code>{joined_extract_result}</code><|im_end|>\n<|im_start|>assistant\n"
        
        output_dict["ground_truth_answer"] = ground_truth_patch
        output_dict["answer"] = ground_truth_patch
        output_dict["target"] = ground_truth_patch
        
        output_dict["commit"] = datum["base_commit"]
        
        print(json.dumps(output_dict), file=f, flush=True)
        
print("Finished processing SWE-bench_oracle.")
print(f"Number of examples that need import: {need_import}")
print(f"Number of remain instances: {num_instances}")
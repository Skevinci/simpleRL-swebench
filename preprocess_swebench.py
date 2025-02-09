import json
import re
from tqdm.auto import tqdm
from datasets import load_dataset 
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "<|im_start|>system\nYou will be provided with a partial code base and an issue statement explaining a problem to resolve. Please reason step by step, and respond with a single patch file that I can apply directly to this repository using git apply in the following format.\n<patch>\n--- a/file.py\n+++ b/file.py\n@@ -1,27 +1,35 @@\n def euclidean(a, b):\n-    while b:\n-        a, b = b, a % b\n-    return a\n+    if b == 0:\n+        return a\n+    return euclidean(b, a % b)\n \n \n def bresenham(x0, y0, x1, y1):\n     points = []\n     dx = abs(x1 - x0)\n     dy = abs(y1 - y0)\n-    sx = 1 if x0 < x1 else -1\n-    sy = 1 if y0 < y1 else -1\n-    err = dx - dy\n+    x, y = x0, y0\n+    sx = -1 if x0 > x1 else 1\n+    sy = -1 if y0 > y1 else 1\n \n-    while True:\n-        points.append((x0, y0))\n-        if x0 == x1 and y0 == y1:\n-            break\n-        e2 = 2 * err\n-        if e2 > -dy:\n+    if dx > dy:\n+        err = dx / 2.0\n+        while x != x1:\n+            points.append((x, y))\n             err -= dy\n-            x0 += sx\n-        if e2 < dx:\n-            err += dx\n-            y0 += sy\n+            if err < 0:\n+                y += sy\n+                err += dx\n+            x += sx\n+    else:\n+        err = dy / 2.0\n+        while y != y1:\n+            points.append((x, y))\n+            err -= dx\n+            if err < 0:\n+                x += sx\n+                err += dy\n+            y += sy\n \n+    points.append((x, y))\n     return points\n</patch>\nPlease put your final patch file within \\boxed{}.<|im_end|>\n<|im_start|>user\n"

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
    
    return prompt


swebench = load_dataset("princeton-nlp/SWE-bench_oracle", split="train")

print(f"Number of examples in SWE-bench_oracle: {len(swebench)}")

need_import = 0
num_instances = 0
with open("/nlp/data/sikaili/simpleRL-reason/train/data/swebench_oracle.jsonl", "w") as f:
    for idx, datum in enumerate(tqdm(swebench, desc="Process SWE-bench_oracle")):
        # if idx > 10:
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
        
        num_instances += 1
        
        # Preprocess the text
        cleaned_input = clean_input(datum["text"])
        processed_input = process_text(cleaned_input)
        output_dict["input"] = f"{processed_input}<|im_end|>\n<|im_start|>assistant"
        
        output_dict["ground_truth_answer"] = ground_truth_patch
        output_dict["answer"] = ground_truth_patch
        output_dict["target"] = ground_truth_patch
        
        output_dict["commit"] = datum["base_commit"]
        
        print(json.dumps(output_dict), file=f, flush=True)
        
print(f"Number of examples that need import: {need_import}")
print(f"Number of remain instances: {num_instances}")
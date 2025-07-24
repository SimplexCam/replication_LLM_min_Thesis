# generate_conversations.py
import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime
import progressbar
from openai import OpenAI
import anthropic
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY

#-------------------------------------API config ------------------------------------------
clientOpenAI = OpenAI(api_key=OPENAI_API_KEY)
clientAnthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

#-------------------------------------Init-------------------------------------------------
SYSTEM_PROMPT = (
    "You are now sharing your thoughts on the question with your partner.\n"
    "You only reply briefly to your thoughts only for a given question."
)

with open("project/themes.json", "r", encoding="utf-8") as f:
    theme_dict = json.load(f)
    THEMES = [theme_dict[str(i)] for i in range(1, 37)]

#--------------------------------------Functions-------------------------------------------
def generate_response(messages, model, temperature=0.7, max_tokens=400, delay=1):
    # Temperature [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    time.sleep(delay)
    if model.startswith("gpt"):
        response = clientOpenAI.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    elif model.startswith("claude"):
        response = clientAnthropic.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text
    else:
        raise ValueError(f"Model {model} not supported.")

def run_conversation(model, seed=42):
    random.seed(seed)
    history = []
    logs = []
    hist=f"{SYSTEM_PROMPT}\n\n"
    
    b = progressbar.ProgressBar(maxval=36)
    b.start()
    for idx, theme in enumerate(THEMES, 1):
        question = f"Question {idx}: {theme}"
        hist=f"{hist}User prompt:\n\t{question}\n"
        
        # --- Agent 1 ---
        messages_agent1 = [{"role": "user", "content": hist}]
        reply_1 = generate_response(messages_agent1, model=model)
        hist=f"{hist}Assistant (First agent):\n\t{reply_1}\n"
        logs.append(("Agent 1", theme, reply_1))

        # --- Agent 2 ---
        messages_agent2 = [{"role": "user", "content": hist}]
        reply_2 = generate_response(messages_agent2, model=model)
        hist=f"{hist}Assistant (Second agent):\n\t{reply_2}\n\n"
        logs.append(("Agent 2", theme, reply_2))
        
        time.sleep(2)
        b.update(idx)
    b.finish()
    return logs, hist

def save_hist_txt(hist, model, seed, output=None):
    filename = output or f"history_{model}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = Path(filename)
    with path.open("w", encoding="utf-8") as f:
        f.write(hist)
    print(f"Historique (txt) enregistré dans : {filename}")
    return str(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, help="Nom du fichier JSON de sortie.")
    args = parser.parse_args()
    
    _, hist = run_conversation(model=args.model, seed=args.seed)
    save_hist_txt(hist, args.model, args.seed, args.output)

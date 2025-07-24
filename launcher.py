# launcher.py
import os
import subprocess
from pathlib import Path

# ============================== HIER KIES JE WAT JE WILT DOEN ==============================

MODEL_TE_LANCEREN = "gpt-4o"  # ← wijzig hier : “gpt-4o”, “claude-opus-4-0”, enz.
SEEDS_TE_GEBRUIKEN = list(range(20))  # ← verander hier seeds dat moet getest worden
CONVERSATIE_GENEREREN = False   # ← True = conversatie genereren
BEOORDELING_MAKEN = True       # ← True = PsychoBench gebruiken

# KIES HIER WELKE SNAPSHOTS JE WILT TESTEN (1 = 12, 2 = 24, 3 = 36)
SNAPSHOT_IDS = [2]

# LIMITEER DE VRAGENLIJSTEN TOT DEZE 8
# GESELECTEERDE_QUESTS = ['BFI', 'DTDD', 'EPQ-R', 'ECR-R', 'CABIN', 'GSE', 'LMS', 'BSRI']
GESELECTEERDE_QUESTS = ['CABIN', 'GSE', 'LMS', 'BSRI']

# ====================================================================================================

CONVERSATIE_DIR = Path("data/conversations")
model_dir = CONVERSATIE_DIR / MODEL_TE_LANCEREN
model_dir.mkdir(parents=True, exist_ok=True)


def launch_conversation(model, seed):
    output_file = f"{CONVERSATIE_DIR}/{MODEL_TE_LANCEREN}/conversation_{model}_seed{seed}.txt"
    if os.path.exists(output_file):
        print(f"[SKIP] {output_file} already exists.")
        return output_file
    print(f"[RUN] Génération de la conversation pour {model}, seed={seed}")
    subprocess.run([
        "python", "generate_conversations.py",
        "--model", model,
        "--seed", str(seed),
        "--output", output_file
    ], check=True)
    return output_file


def launch_evaluation(model, seed, json_file, snapshot_id):
    name_exp = f"{model}_seed{seed}_snapshot{snapshot_id}"
    print(f"[RUN] Évaluation PsychoBench pour {name_exp}: {snapshot_id}")
    subprocess.run([
        "python", "evaluate_conversations.py",
        "--model", model,
        "--seed", str(seed),
        "--input", json_file,
        "--name_exp", name_exp,
        "--snapshot_id", str(snapshot_id),
        "--questionnaire", ','.join(GESELECTEERDE_QUESTS)
    ], check=True)


if __name__ == "__main__":
    for seed in SEEDS_TE_GEBRUIKEN:
        json_file = None

        if CONVERSATIE_GENEREREN:
            json_file = launch_conversation(MODEL_TE_LANCEREN, seed)

        if BEOORDELING_MAKEN:
            if json_file is None:
                json_file = f"{CONVERSATIE_DIR}/{MODEL_TE_LANCEREN}/conversation_{MODEL_TE_LANCEREN}_seed{seed}.txt"
            for snapshot_id in SNAPSHOT_IDS:
                launch_evaluation(MODEL_TE_LANCEREN, seed, json_file, snapshot_id)

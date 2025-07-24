import argparse
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'PsychoBench')))
from run_psychobench import run_psychobench
from generator import generator
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY


def extract_convo_snapshots_from_text(convo_text, checkpoints=["User prompt:\n\tQuestion 13", "User prompt:\n\tQuestion 25"]):
    snapshots = {}
    
    for i, stop_phrase in enumerate(checkpoints):
        cutoff = convo_text.find(stop_phrase)
        if cutoff != -1:
            snapshots[i + 1] = convo_text[:cutoff].strip()
        else:
            print(f"[WARN] {stop_phrase} non trouvé dans le texte.")
            snapshots[i + 1] = convo_text.strip()

    # Ajoute la version complète (36 questions)
    snapshots[len(checkpoints) + 1] = convo_text.strip()

    return snapshots


def run_evaluation(convo_text, model, snapshot_id, seed, name_exp, questionnaire):
    if model.startswith("gpt"):
        args = argparse.Namespace(
            model=model,
            questionnaire=questionnaire,
            AI_key=OPENAI_API_KEY,
            shuffle_count=9,
            test_count=10,
            name_exp=name_exp,
            significance_level=0.01,
            mode="auto",
            testing_file=None,
            prompt_context=convo_text
        )
    elif model.startswith("claude"):
        args = argparse.Namespace(
            model=model,
            questionnaire=questionnaire,
            AI_key=ANTHROPIC_API_KEY,
            shuffle_count=9,
            test_count=10,
            name_exp=name_exp,
            significance_level=0.01,
            mode="auto",
            testing_file=None,
            prompt_context=convo_text
        )
    else:
        raise ValueError(f"Model {model} not supported.")

    run_psychobench(args, generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Fichier texte avec la conversation.")
    parser.add_argument("--model", required=True, help="Nom du modèle à réévaluer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name_exp", type=str, default=None, help="Nom de l'expérience pour sauvegarde.")
    parser.add_argument("--snapshot_id", type=int, choices=[1, 2, 3], default=None,
                        help="Quel snapshot évaluer (1 = 12, 2 = 24, 3 = 36). Si vide, évalue les trois.")
    parser.add_argument("--questionnaire", type=str, default="ALL", help="Liste des questionnaires à utiliser")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        convo_text = f.read()

    snapshots = extract_convo_snapshots_from_text(convo_text)

    if args.snapshot_id:  # Un seul snapshot à lancer
        convo = snapshots[args.snapshot_id]
        run_evaluation(convo, args.model, args.snapshot_id, args.seed, args.name_exp, args.questionnaire)
    else:  # Tous les snapshots
        for snapshot_id, convo in snapshots.items():
            name_exp = f"{args.model}_seed{args.seed}_snapshot{snapshot_id}"
            run_evaluation(convo, args.model, snapshot_id, args.seed, name_exp, args.questionnaire)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict
from math import sqrt

# Mapping CABIN categories to RIASEC --> Otherwise 41 subcategories
cabin_to_riasec = {
    "Accounting": "Conventional", "Agriculture": "Realistic", "Animal Service": "Social", "Applied Arts and Design": "Artistic",
    "Athletics": "Realistic", "Business Iniatives": "Enterprising", "Construction/WoodWork": "Realistic",
    "Culinary Art": "Realistic", "Engineering": "Realistic", "Finance": "Conventional", "Health Care Service": "Social",
    "Human Resources": "Enterprising", "Humanities": "Artistic", "Information Technology": "Conventional",
    "Law": "Enterprising", "Life Science": "Investigative", "Management/Administration": "Enterprising",
    "Marketing/Advertising": "Enterprising", "Mathematics/Statistics": "Investigative", "Mechanics/Electronics": "Realistic",
    "Media": "Artistic", "Medical Science": "Investigative", "Music": "Artistic", "Nature/Outdoors": "Realistic",
    "Office Work": "Conventional", "Performing Arts": "Artistic", "Personal Service": "Social",
    "Physical Science": "Investigative", "Physical/Manual Labor": "Realistic", "Politics": "Enterprising",
    "Professional Advising": "Social", "Protective Service": "Realistic", "Public Speaking": "Enterprising",
    "Religious Activities": "Social", "Sales": "Enterprising", "Social Science": "Investigative",
    "Social Service": "Social", "Teaching/Education": "Social", "Transportation/Machine Operation": "Realistic",
    "Visual Arts": "Artistic", "Writing": "Artistic"
}

def map_cabin_categories(df):
    mask = df["Questionnaire"] == "CABIN"
    df.loc[mask, "Category"] = df.loc[mask, "Category"].map(cabin_to_riasec)
    return df

# Order and renaming to match article's names and order for comparison
questionnaire_category_order = [
    ("BFI", "Openness"),
    ("BFI", "Conscientiousness"),
    ("BFI", "Extraversion"),
    ("BFI", "Agreeableness"),
    ("BFI", "Neuroticism"),
    ("EPQ-R", "Extraversion"),
    ("EPQ-R", "Psychoticism"),
    ("EPQ-R", "Neuroticism"),
    ("EPQ-R", "Lying"),
    ("DTDD", "Machiavellianism"),
    ("DTDD", "Psychopathy"),
    ("DTDD", "Narcissism"),
    ("BSRI", "Masculine"),
    ("BSRI", "Feminine"),
    ("CABIN", "Realistic"),
    ("CABIN", "Investigative"),
    ("CABIN", "Artistic"),
    ("CABIN", "Social"),
    ("CABIN", "Enterprising"),
    ("CABIN", "Conventional"),
    ("ECR-R", "Attachment-related Anxiety"),
    ("ECR-R", "Attachment-related Avoidance"),
    ("GSE", "Overall"),
    ("LMS", "Factor rich"),
    ("LMS", "Factor motivator"),
    ("LMS", "Factor important")
]

#Values from the article
article_values = {
    ("BFI", "Openness"): "⬇️",
    ("BFI", "Conscientiousness"): "⬇️",
    ("BFI", "Extraversion"): "⬇️",
    ("BFI", "Agreeableness"): "⬇️",
    ("BFI", "Neuroticism"): "⬇️",
    ("EPQ-R", "Extraversion"): "⬇️",
    ("EPQ-R", "Psychoticism"): "⬇️",
    ("EPQ-R", "Neuroticism"): "⬇️",
    ("EPQ-R", "Lying"): "⬇️",
    ("DTDD", "Machiavellianism"): "⬇️",
    ("DTDD", "Psychopathy"): "⬇️",
    ("DTDD", "Narcissism"): "🔀",
    ("BSRI", "Masculine"): "➖",
    ("BSRI", "Feminine"): "⬇️",
    ("CABIN", "Realistic"): "⬇️",
    ("CABIN", "Investigative"): "🔀",
    ("CABIN", "Artistic"): "➖",
    ("CABIN", "Social"): "⬆️",
    ("CABIN", "Enterprising"): "➖",
    ("CABIN", "Conventional"): "🔀",
    ("ECR-R", "Attachment-related Anxiety"): "⬇️",
    ("ECR-R", "Attachment-related Avoidance"): "⬇️",
    ("GSE", "Overall"): "➖",
    ("LMS", "Factor rich"): "⬇️",
    ("LMS", "Factor motivator"): "⬇️",
    ("LMS", "Factor important"): "⬇️"
}

def apply_sorting(df_pivot):
    df_pivot["Order"] = df_pivot.apply(
        lambda row: questionnaire_category_order.index((row["Questionnaire"], row["Category"]))
        if (row["Questionnaire"], row["Category"]) in questionnaire_category_order else float('inf'), axis=1)
    return df_pivot.sort_values("Order").drop(columns="Order")

def parse_md_tables(md_path):
    results = []
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    n = None
    for line in lines:
        if "gpt-4o" in line and "n =" in line:
            match_n = re.search(r"n\s*=\s*(\d+)", line)
            if match_n:
                n = int(match_n.group(1))
            break

    for line in lines:
        if line.startswith("|") and "$\\pm$" in line and not line.lower().startswith("| category"):
            parts = [p.strip() for p in line.strip().split("|") if p.strip()]
            if len(parts) >= 2:
                category = parts[0]
                stat_match = re.match(r"([\d\.]+)\s*\$\\pm\$[\s]*([\d\.]+)", parts[1])
                if stat_match:
                    mean = float(stat_match.group(1))
                    std = float(stat_match.group(2))
                    results.append((category, mean, std, n))
    return results

def pooled_std(values):
    total_n = sum(v[2] for v in values if v[2] is not None)
    if total_n <= 1:
        return 0.0
    weighted_mean = sum(v[0] * v[2] for v in values if v[2] is not None) / total_n
    ss_total = sum(
        (v[2] - 1) * v[1]**2 + v[2] * (v[0] - weighted_mean)**2
        for v in values if v[2] is not None
    )
    return sqrt(ss_total / (total_n - 1))

def analyze_and_generate_trend_table(base_path="results", model="gpt-4o"):
    base_path = Path(base_path)
    all_md_files = sorted(base_path.glob(f"{model}_seed*_snapshot*-*.md"))
    print(f"\U0001F4C1 {len(all_md_files)} .md-bestanden gedetecteerd")

    grouped = defaultdict(list)
    for path in all_md_files:
        match = re.match(rf"{model}_seed(\d+)_snapshot(\d+)-(.+)\.md", path.name)
        if not match:
            print(f"❌ Bestandsnaam genegeerd: {path.name}")
            continue
        seed, snapshot, questionnaire = match.groups()
        snapshot = int(snapshot)
        questionnaire = questionnaire.strip().upper()
        parsed = parse_md_tables(path)
        for category, mean, std, n in parsed:
            if questionnaire == "EPQ-R" and category == "Pschoticism":
                category = "Psychoticism"  # correction
            if questionnaire == "DTDD" and category == "Neuroticism":
                category = "Narcissism"  # correction
            if questionnaire == "CABIN":
                cleaned = category.strip().replace("–", "-").replace("’", "'")
                riasec = cabin_to_riasec.get(cleaned)
                if riasec is not None:
                    grouped[(questionnaire, riasec, snapshot)].append((mean, std, n))
                else:
                    print(f"⚠️ Geen mapping voor categorie CABIN: '{category}' dans {path.name}")
                    grouped[(questionnaire, None, snapshot)].append((mean, std, n))

            else:
                grouped[(questionnaire, category, snapshot)].append((mean, std, n))
        print(f"📥 Toegevoegde gegevens : Questionnaire={questionnaire}, Snapshot={snapshot}, Catégorie(s)={[cat for cat, _, _, _ in parsed]}")

    summary = []
    for (questionnaire, category, snapshot), values in grouped.items():
        if category is None:
            continue  # Lijnen zonder RIASEC-overeenkomst negeren
        ns = [v[2] for v in values if v[2] is not None]
        total_n = sum(ns)
        if total_n == 0:
            continue
        global_mean = sum(v[0] * v[2] for v in values if v[2] is not None) / total_n
        global_std = pooled_std(values)
        summary.append({
            "Questionnaire": questionnaire,
            "Category": category,
            "Snapshot": snapshot,
            "Mean": global_mean,
            "Std": global_std,
            "Total N": total_n
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(base_path / f"{model}_aggregated_md_summary.csv", index=False)
    print(f"✅ CSV geschreven : {base_path / f'{model}_aggregated_md_summary.csv'}")


    duplicates = df_summary.duplicated(subset=["Questionnaire", "Category", "Snapshot"], keep=False)
    if duplicates.any():
        print("\n⚠️ Duplicaten gedetecteerd voor tripletten (Questionnaire, Category, Snapshot):")
        print(df_summary[duplicates].sort_values(["Questionnaire", "Category", "Snapshot"]))
        
    df_summary_grouped = df_summary.groupby(["Questionnaire", "Category", "Snapshot"], as_index=False).agg({
        "Mean": "mean",
        "Std": "mean",
        "Total N": "sum"
    })

    df_summary_grouped["Snapshot"] = df_summary_grouped["Snapshot"].astype(int)

    df_pivot = df_summary_grouped.pivot(index=["Questionnaire", "Category"], columns="Snapshot", values="Mean").reset_index()
    df_pivot.columns.name = None

    snapshot_map = {1: "S12", 2: "S24", 3: "S36"}
    df_pivot.columns = [snapshot_map.get(int(c), c) if str(c).isdigit() else c for c in df_pivot.columns]

    for col1, col2, new_col in [("S12", "S24", "D12_24"), ("S24", "S36", "D24_36"), ("S12", "S36", "D12_36")]:
        df_pivot[new_col] = df_pivot.apply(
            lambda row: row[col2] - row[col1] if pd.notna(row.get(col1)) and pd.notna(row.get(col2)) else np.nan,
            axis=1
        )

    def classify_trend(row):
        delta12_24 = row.get("D12_24")
        delta24_36 = row.get("D24_36")
        delta12_36 = row.get("D12_36")
        signs = [np.sign(delta12_24), np.sign(delta24_36), np.sign(delta12_36)]
        if all(s == 1 for s in signs):
            return "⬆️"
        elif all(s == -1 for s in signs):
            return "⬇️"
        elif all(s == 0 for s in signs):
            return "➖"
        else:
            return "🔀"

    df_pivot["Trend"] = df_pivot.apply(
        lambda row: classify_trend(row) if pd.notna(row.get("D12_24")) and pd.notna(row.get("D24_36")) and pd.notna(row.get("D12_36")) else "",
        axis=1
    )

    df_pivot = apply_sorting(df_pivot)
    
    df_custom = pd.DataFrame([
        {"Questionnaire": q, "Category": c, "Trend article": v}
        for (q, c), v in article_values.items()
        ])
    
    df_pivot = df_pivot.merge(df_custom, on=["Questionnaire", "Category"], how="left")
    
    def shorten_category(questionnaire, category):
        if questionnaire == "ECR-R":
            if "Anxiety" in category:
                return "Attachment Anxiety"
            elif "Avoidance" in category:
                return "Attachment Avoidance"
        return category


    def convert_arrow_to_emoji_cmd(symbol):
        return {
            "⬆️": r"\emoji{up-right-arrow}",
            "⬇️": r"\emoji{down-right-arrow}",
            "➖": r"\emoji{minus}",
            "🔀": r"\emoji{shuffle-tracks-button}"
        }.get(symbol, symbol)

    latex = """\\begin{tabular}{|l|l|c|c|c|c|c|c|c|c|}\n\\hline\nVragenlijst & Categorie & $S_{12}$ & $S_{24}$ & $S_{36}$ & $\\Delta_{12-24}$ & $\\Delta_{24-36}$ & $\\Delta_{12-36}$ & Trend & $Trend_{article}$\\\\ \n\\hline\n"""
    for _, row in df_pivot.iterrows():
        def f(v): return f"{v:.2f}" if pd.notna(v) else "–"
        category_label = shorten_category(row['Questionnaire'], row['Category'])
        latex += f"{row['Questionnaire']} & {category_label} & {f(row.get('S12'))} & {f(row.get('S24'))} & {f(row.get('S36'))} & {f(row.get('D12_24'))} & {f(row.get('D24_36'))} & {f(row.get('D12_36'))} & {convert_arrow_to_emoji_cmd(row['Trend'])} & {convert_arrow_to_emoji_cmd(row.get('Trend article', '-'))}\\\\ \n"
    latex += "\\hline\n\\end{tabular}"

    out_path = base_path / f"{model}_trend_arrows_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"📄 Vergelijkende tabel opgeslagen: {out_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    existing_cols = [col for col in ["S12", "S24", "S36"] if col in df_pivot.columns]
    plot_df = df_pivot.dropna(subset=existing_cols)

    for idx, row in plot_df.iterrows():
        x, y = [], []
        if pd.notna(row.get("S12")): x.append(12); y.append(row["S12"])
        if pd.notna(row.get("S24")): x.append(24); y.append(row["S24"])
        if pd.notna(row.get("S36")): x.append(36); y.append(row["S36"])
        if len(x) >= 2:
            label = f"{row['Questionnaire']} – {row['Category']}"
            ax.plot(x, y, marker="o", label=label)

    ax.set_title("Ontwikkeling van de gemiddelde scores per categorie")
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("Gemiddelde score")
    ax.set_xticks([12, 24, 36])
    ax.set_ylim(0, 5)
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()

    trend_plot_path = base_path / f"{model}_trend_lines.png"
    plt.savefig(trend_plot_path)
    plt.close()
    print(f"📈 Grafiek van geregistreerde trends: {trend_plot_path}")

    excel_path = Path(base_path) / f"{model}_trend_table.xlsx"
    try:
        df_pivot.to_excel(excel_path, index=False)
        print(f"💾 Excel opgeslagen: {excel_path}")
    except PermissionError:
        print(f"❌ Het is niet mogelijk om in het Excel-bestand te schrijven. Is het geopend?")

    missing_snapshots = df_pivot[
        df_pivot[["S12", "S24", "S36"]].isna().any(axis=1)
    ][["Questionnaire", "Category", "S12", "S24", "S36"]]

    if not missing_snapshots.empty:
        print("\n⚠️ Lijnen met ontbrekende snapshots:")
        print(missing_snapshots.to_string(index=False))
    else:
        print("\n✅ Alle lijnen hebben S12, S24 en S36.")


if __name__ == "__main__":
    analyze_and_generate_trend_table()

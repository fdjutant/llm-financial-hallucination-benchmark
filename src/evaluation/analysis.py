import pandas as pd
import os
import glob

def compute_metrics(df):
    """
    Compute derived metrics for a DataFrame of LLM results.
    Adds columns: rag_correct, knowledge_correct, hallucinated, adversarial_correct
    """
    from .llm_interface import compare_answers
    def _hallucinated(row):
        answer = str(row.get('answer_knowledge', ''))
        correct = row['knowledge_correct']
        return (
            not correct and
            answer.strip() != '' and
            'unknown' not in answer.lower() and
            'n/a' not in answer.lower()
        )
    def _adversarial_correct(row):
        gt = str(row.get('ground_truth', ''))
        ans = str(row.get('answer_adversarial', ''))
        return gt in ans
    # Compute metrics
    df['rag_correct'] = df.apply(lambda row: compare_answers(row.get('answer_rag', ''), row.get('ground_truth', '')), axis=1)
    df['knowledge_correct'] = df.apply(lambda row: compare_answers(row.get('answer_knowledge', ''), row.get('ground_truth', '')), axis=1)
    df['hallucinated'] = df.apply(_hallucinated, axis=1)
    df['adversarial_correct'] = df.apply(_adversarial_correct, axis=1)
    return df

def analyze_results(folders=['./data/results/debug/']):
    """
    Print research-grade analysis.
    Accepts:
      - folders: path or list of paths to folders containing CSVs (default: ['./data/results/debug/'])
    """
    if isinstance(folders, str):
        folders = [folders]

    csv_files = []
    for folder in folders:
        if not os.path.isdir(folder):
            raise ValueError(f"Invalid folder path: {folder}")
        csv_files.extend(glob.glob(os.path.join(folder, '*_results.csv')))

    if not csv_files:
        print("No CSV files found for analysis.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df = compute_metrics(df)

    print("HALLUCINATION BENCHMARK RESULTS")
    
    summary_rows = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        rag_acc = model_data['rag_correct'].mean() * 100
        knowledge_acc = model_data['knowledge_correct'].mean() * 100
        halluc_rate = model_data['hallucinated'].mean() * 100
        adversarial_acc = model_data['adversarial_correct'].mean() * 100
        summary_rows.append({
            'model': model,
            'n': len(model_data),
            'rag_acc': rag_acc,
            'knowledge_acc': knowledge_acc,
            'halluc_rate': halluc_rate,
            'adversarial_acc': adversarial_acc,
            'halluc_index': halluc_rate - (100 - knowledge_acc)
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print summary as aligned ASCII table
    print("\n" + "-"*100)
    header = f"{'Model':<35} {'N':>6} {'RAG Acc':>10} {'Knowl Acc':>10} {'Halluc %':>10} {'Adv Acc':>10} {'HallucIdx':>10}"
    print(header)
    print("-"*100)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<35} {row['n']:6d} {row['rag_acc']:10.1f} {row['knowledge_acc']:10.1f} {row['halluc_rate']:10.1f} {row['adversarial_acc']:10.1f} {row['halluc_index']:10.1f}")
    print("-"*100)

    return summary_df
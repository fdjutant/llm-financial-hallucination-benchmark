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

def analyze_results(folder='./data/results/debug/'):
    """
    Print research-grade analysis.
    Accepts:
      - folder: path to a folder containing CSVs (default: './data/results/debug/')
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Invalid folder path: {folder}")

    csv_files = glob.glob(os.path.join(folder, '*_results.csv'))

    if not csv_files:
        print("No CSV files found for analysis.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df = compute_metrics(df)

    print("\n" + "="*70)
    print("HALLUCINATION BENCHMARK RESULTS")
    print("="*70)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        rag_acc = model_data['rag_correct'].mean() * 100
        knowledge_acc = model_data['knowledge_correct'].mean() * 100
        halluc_rate = model_data['hallucinated'].mean() * 100
        adversarial_acc = model_data['adversarial_correct'].mean() * 100
        print(f"\n{model} (n = {len(model_data)})")
        print(f"  RAG Accuracy:          {rag_acc:6.1f}% (with context)")
        print(f"  Knowledge Accuracy:    {knowledge_acc:6.1f}% (no context)")
        print(f"  Hallucination Rate:    {halluc_rate:6.1f}% (confident wrong answers)")
        print(f"  Adversarial Robustness: {adversarial_acc:6.1f}% (trusts correct source)")
        print(f"  Hallucination Index:   {(halluc_rate - (100-knowledge_acc)):+.1f}%")
    
    print("\n" + "="*70)

    # return df
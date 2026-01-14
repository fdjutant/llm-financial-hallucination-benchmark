import pandas as pd
import os
import glob

def compute_metrics(df):
    """
    Compute derived metrics for a DataFrame of LLM results.
    Adds columns: rag_correct, knowledge_correct, hallucinated_rag, hallucinated_knowledge, adversarial_correct
    """
    from .llm_interface import compare_answers

    # Helper function to determine if an answer is hallucinated
    def _hallucinated(answer, correct):
        """
        A hallucinated answer is one that is incorrect, non-empty, and does not contain
        placeholders like 'unknown' or 'n/a'.
        """
        answer = str(answer)
        return (
            not correct and
            answer.strip() != '' and
            'unknown' not in answer.lower() and
            'n/a' not in answer.lower()
        )

    # Helper function to determine if the adversarial answer is correct
    def _adversarial_correct(row):
        """
        An adversarial answer is correct if the ground truth is present in the adversarial answer.
        """
        gt = str(row.get('ground_truth', ''))
        ans = str(row.get('answer_adversarial', ''))
        return gt in ans

    # Compute metrics
    # RAG (Retrieval-Augmented Generation) correctness: Checks if the RAG answer matches the ground truth
    df['rag_correct'] = df.apply(lambda row: compare_answers(row.get('answer_rag', ''), row.get('ground_truth', '')), axis=1)
    
    # Knowledge correctness: Checks if the knowledge-based answer matches the ground truth
    df['knowledge_correct'] = df.apply(lambda row: compare_answers(row.get('answer_knowledge', ''), row.get('ground_truth', '')), axis=1)
    
    # Hallucination in RAG: Checks if the RAG answer is hallucinated
    df['hallucinated_rag'] = df.apply(lambda row: _hallucinated(row.get('answer_rag', ''), row['rag_correct']), axis=1)
    
    # Hallucination in knowledge-based answers: Checks if the knowledge-based answer is hallucinated
    df['hallucinated_knowledge'] = df.apply(lambda row: _hallucinated(row.get('answer_knowledge', ''), row['knowledge_correct']), axis=1)
    
    # Adversarial correctness: Checks if the adversarial answer contains the ground truth
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
        halluc_rate_rag = model_data['hallucinated_rag'].mean() * 100
        halluc_rate_knowledge = model_data['hallucinated_knowledge'].mean() * 100
        adversarial_acc = model_data['adversarial_correct'].mean() * 100

        # Hallucination Index for RAG: Measures the gap between hallucination rate and the complement of accuracy
        halluc_index_rag = halluc_rate_rag - (100 - rag_acc)

        # Hallucination Index for Knowledge: Measures the gap between hallucination rate and the complement of accuracy
        halluc_index_knowledge = halluc_rate_knowledge - (100 - knowledge_acc)

        summary_rows.append({
            'model': model,
            'n': len(model_data),
            'rag_acc': rag_acc,
            'knowledge_acc': knowledge_acc,
            'halluc_rate_rag': halluc_rate_rag,
            'halluc_rate_knowledge': halluc_rate_knowledge,
            'adversarial_acc': adversarial_acc,
            'halluc_index_rag': halluc_index_rag,
            'halluc_index_knowledge': halluc_index_knowledge
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print summary as aligned ASCII table
    print("\n" + "-"*90)
    header = (
        f"{'Model':<28} {'N':>6} "
        f"{'RAG Acc (%)':>12} {'HallRAG (%)':>12} "
        f"{'Knowl Acc (%)':>14} {'HallKnowl (%)':>14} "
        f"{'Adv Acc (%)':>12}"
    )
    print(header)
    print("-"*90)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<28} {row['n']:6d} "
            f"{row['rag_acc']:12.1f} {row['halluc_rate_rag']:12.1f} "
            f"{row['knowledge_acc']:14.1f} {row['halluc_rate_knowledge']:14.1f} "
            f"{row['adversarial_acc']:12.1f}"
        )
    print("-"*90)

    return summary_df
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
        An empty answer is not treated as hallucinated.
        """
        answer = str(answer).strip()
        return (
            not correct and
            answer != '' and
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

def analyze_results(base_folder='./data/results/llm_batch/'):
    """
    Print research-grade analysis.
    Looks for CSV files inside subfolders of the specified base folder.
    Accepts:
      - base_folder: path to the base folder containing subfolders with CSVs (default: './data/results/llm_batch/')
    """
    if not os.path.isdir(base_folder):
        raise ValueError(f"Invalid base folder path: {base_folder}")

    # Look for CSV files in subfolders of the base folder
    csv_files = glob.glob(os.path.join(base_folder, '*', '*_results.csv'))

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
        f"{'Model':<38} {'N':>6} "
        f"{'RAG Acc (%)':>12} {'HallRAG (%)':>12} "
        f"{'Knowl Acc (%)':>14} {'HallKnowl (%)':>14} "
        f"{'Adv Acc (%)':>12}"
    )
    print(header)
    print("-"*90)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<38} {row['n']:6d} "
            f"{row['rag_acc']:12.1f} {row['halluc_rate_rag']:12.1f} "
            f"{row['knowledge_acc']:14.1f} {row['halluc_rate_knowledge']:14.1f} "
            f"{row['adversarial_acc']:12.1f}"
        )
    print("-"*90)

    return summary_df

def fixed_missing_columns_in_mistral_results(ans_csv_path, qa_csv_path):

    # Load the CSV files
    ans_df = pd.read_csv(ans_csv_path)
    qa_df = pd.read_csv(qa_csv_path)
    
    # Replace columns from ans_df with corresponding columns from qa_df
    merged_df = ans_df.drop(columns=['question', 'entity', 'year',
                                     'metric', 'segment', 'ground_truth']).merge(
        qa_df[['id', 'generated_question', 'entity_name', 'year',
               'original_metric', 'segment', 'ground_truth_value']], 
        on='id', 
        how='left'
    )
    
    # Rename columns to match original naming
    merged_df = merged_df.rename(columns={
        'generated_question': 'question',
        'entity_name': 'entity',
        'original_metric': 'metric',
        'ground_truth_value': 'ground_truth'
    })
    
    return merged_df
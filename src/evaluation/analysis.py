import pandas as pd

def analyze_results(results):
    """Print research-grade analysis."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("HALLUCINATION BENCHMARK RESULTS")
    print("="*80)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        rag_acc = model_data['rag_accuracy'].mean() * 100
        knowledge_acc = model_data['knowledge_accuracy'].mean() * 100
        halluc_rate = model_data['hallucinated'].mean() * 100
        adversarial_acc = model_data['adversarial_correct'].mean() * 100
        
        print(f"\n{model}:")
        print(f"  RAG Accuracy:          {rag_acc:6.1f}% (with context)")
        print(f"  Knowledge Accuracy:    {knowledge_acc:6.1f}% (no context)")
        print(f"  Hallucination Rate:    {halluc_rate:6.1f}% (confident wrong answers)")
        print(f"  Adversarial Robustness: {adversarial_acc:6.1f}% (trusts correct source)")
        print(f"  Hallucination Index:   {(halluc_rate - (100-knowledge_acc)):+.1f}%")
    
    print("\n" + "="*80)
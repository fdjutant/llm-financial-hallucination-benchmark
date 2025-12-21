from pathlib import Path
from openai import OpenAI
import pandas as pd
import json
import os

# Set up API keys (for different providers)
openai_api_key = Path(Path(__file__).resolve().parents[2]/"OPENAI_API_KEY").read_text().strip()
groq_api_key = Path(Path(__file__).resolve().parents[2]/"GROQ_API_KEY").read_text().strip()

def get_client_for_model(model_name):
    """
    Returns the appropriate client based on the model provider.
    """
    if model_name.startswith("gpt-"):
        # OpenAI models
        return OpenAI(api_key=openai_api_key), "openai"
    
    elif model_name.startswith("llama-") or model_name.startswith("openai"):
        # Groq models
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set. Sign up at https://console.groq.com")
        return OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        ), "groq"

def evaluate_models_on_qa_pairs(qa_pairs_path, evaluation_output_path):
    """
    Test multiple models (OpenAI + Groq) against QA pairs.
    """
    
    df = pd.read_csv(qa_pairs_path)
    print(f"Loaded {len(df)} QA pairs")
    
    # Mix of OpenAI and Groq models
    models_to_test = [
        # OpenAI (paid)
        # "gpt-4o",
        "gpt-4o-mini",
        
        # Groq (FREE - 14,400 requests/day)
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        # "meta-llama/llama-guard-4-12b"
    ]
    
    results = []
    
    for idx, row in df.iterrows():
        question = row['generated_question']
        ground_truth_value = row['ground_truth_value']
        entity = row['entity_name']
        year = row['year']
        
        for model in models_to_test:
            try:
                client, provider = get_client_for_model(model)
                
                prompt = (
                    f"You are a financial analyst.\n"
                    f"Answer this question as accurately as possible:\n\n"
                    f"Q: {question}\n\n"
                    "Respond in JSON: {\"answer\": \"...\", \"confidence\": 0-100}"
                )
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst. Output valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=200,
                    response_format={"type": "json_object"} if provider == "openai" else None
                )
                
                content = response.choices[0].message.content
                # Handle Groq which might not return pure JSON
                if "```" in content:
                    content = content.split("```json")[1].split("```")[0]
                
                parsed = json.loads(content)
                model_answer = str(parsed.get("answer", ""))
                
                results.append({
                    "entity": entity,
                    "year": year,
                    "question": question,
                    "ground_truth": ground_truth_value,
                    "model": model,
                    "provider": provider,
                    "model_answer": model_answer,
                    "confidence": parsed.get("confidence", "N/A")
                })
                
                print(f"✓ {model} - Row {idx}")
                
            except Exception as e:
                print(f"✗ {model} - Row {idx}: {e}")
                results.append({
                    "entity": entity,
                    "year": year,
                    "question": question,
                    "ground_truth": ground_truth_value,
                    "model": model,
                    "provider": "unknown",
                    "model_answer": "ERROR",
                    "confidence": "N/A"
                })
        
        if (idx + 1) % 5 == 0:
            print(f"\n--- Evaluated {idx + 1}/{len(df)} QA pairs ---\n")
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(evaluation_output_path, index=False)
        
    return output_df
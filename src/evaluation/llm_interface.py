from pathlib import Path
from openai import OpenAI
import pandas as pd
import json
import os
import ast
import re

# Set up API keys (for different providers)
openai_api_key = Path(Path(__file__).resolve().parents[2]/
                      "API_KEY"/"OPENAI_API_KEY").read_text().strip()
groq_api_key = Path(Path(__file__).resolve().parents[2]/
                   "API_KEY"/"GROQ_API_KEY").read_text().strip()
google_api_key = Path(Path(__file__).resolve().parents[2]/
                   "API_KEY"/"GOOGLE_API_KEY").read_text().strip()

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
    
    elif model_name.startswith("gemini-"):
        # Gemini models
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        return OpenAI(
            api_key=google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai"
        ), "gemini"

def evaluate_knowledge_base(qa_pairs_path, evaluation_output_path):
    """
    Test multiple models (OpenAI + Groq + Gemini) against QA pairs.
    """
    
    df = pd.read_csv(qa_pairs_path)
    print(f"Loaded {len(df)} QA pairs")
    
    # Mix of OpenAI, Groq, and Gemini models
    models_to_test = [
        # OpenAI (paid)
        # "gpt-4o-mini",
        
        # Groq (FREE - 14,400 requests/day)
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        
        # Gemini
        "gemini-2.5-flash"
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
                try:
                    # Attempt to parse the response as JSON
                    if "```" in content:
                        content = content.split("```json")[1].split("```")[0]
                    parsed = json.loads(content)
                    model_answer = str(parsed.get("answer", ""))
                    confidence = parsed.get("confidence", "N/A")
                except json.JSONDecodeError:
                    # If parsing fails, use the raw content as the answer
                    model_answer = content.strip()
                    confidence = "N/A"

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

def evaluate_with_xbrl_context(qa_pairs_path, evaluation_output_path):

    df = pd.read_csv(qa_pairs_path)
    print(f"Loaded {len(df)} QA pairs")
    
    # Mix of OpenAI, Groq, and Gemini models
    models_to_test = [
        # OpenAI (paid)
        # "gpt-4o-mini",
        
        # Groq (FREE - 14,400 requests/day)
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        
        # Gemini
        # "gemini-2.5-flash"
    ]
    
    results = []
    
    for idx, row in df.iterrows():
        question = row['generated_question']
        ground_truth_value = row['ground_truth_value']
        entity = row['entity_name']
        year = row['year']
        metric = row['original_metric']
        segment = row['segment']
        
        # Create a mini "document" from your XBRL data
        if segment == "Income_Statement":
            document = (
                f"CONSOLIDATED INCOME STATEMENT ({entity}, {year})\n"
                f"{metric}: {ground_truth_value}\n"
            )
        elif segment == "Balance_Sheet":
            document = (
                f"CONSOLIDATED BALANCE SHEET ({entity}, {year})\n"
                f"{metric}: {ground_truth_value}\n"
            )
        elif segment == "Cash_Flow":
            document = (
                f"CONSOLIDATED STATEMENT OF CASH FLOWS ({entity}, {year})\n"
                f"{metric}: {ground_truth_value}\n"
            )
        elif segment == "Company_Specific_Metric":
            document = (
                f"CONSOLIDATED STATEMENT OF CASH FLOWS ({entity}, {year})\n"
                f"{metric}: {ground_truth_value}\n"
            )
        else:
            document = f"{metric}: {ground_truth_value}\n"
        
        # print(document)
        for model in models_to_test:
            try:
                client, provider = get_client_for_model(model)
                
                prompt = (
                f"You are a financial analyst reading an annual report excerpt.\n\n"
                f"FINANCIAL STATEMENT:\n"
                f"{document}\n"
                f"Based ONLY on the financial statement above, answer the question.\n"
                f"CONFIDENCE SCALE:\n"
                f"90-100: Well-known figure. Example: 'Apple's 2023 revenue was ~$383B' (conf: 95)\n"
                f"70-89:  Confident in this knowledge. Example: 'EBITDA = Earnings Before Interest...' (conf: 85)\n"
                f"50-69:  Reasonable inference but may be slightly off. Example: 'GSK 2023 revenue ~£30B' (conf: 65)\n"
                f"30-49:  Educated guess. Example: 'Typical biotech burn rate $5-10M/year' (conf: 40)\n"
                f"0-29:   Very uncertain - if this low, respond UNKNOWN instead (conf: 15)\n\n"
                f"QUESTION: {question}\n"
                f"RESPONSE: {{'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}"
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
                llm_answer = content.strip()
                answer, confidence, reasoning = robust_extract_json(llm_answer)

                results.append({
                    "entity": entity,
                    "year": year,
                    "question": question,
                    "ground_truth": ground_truth_value,
                    "model": model,
                    "provider": provider,
                    "model_answer": answer,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "full_answer": llm_answer
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
                    "confidence": "N/A",
                    "reasoning": "N/A",
                    "full_answer": "ERROR"
                })
        
        if (idx + 1) % 5 == 0:
            print(f"\n--- Evaluated {idx + 1}/{len(df)} QA pairs ---\n")
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(evaluation_output_path, index=False)
        
    return output_df

def robust_extract_json(text):
    """
    Extracts answer, confidence, and reasoning from messy, mixed-format JSON strings.
    Handles:
      - CSV escaped quotes (""key"": ""value"")
      - Single quotes ({'key': 'value'})
      - Truncated strings
      - Numeric formatting with commas
    """
    if not isinstance(text, str):
        return {"answer": "", "confidence": "N/A", "reasoning": ""}

    text = text.strip()
    
    # 1. Cleaning: Fix CSV double-double quotes and outer wrappers
    # If the whole string is wrapped in quotes like "{...}", remove them
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
        
    # Replace CSV escape sequence "" with normal "
    clean_text = text.replace('""', '"')
    
    parsed = None
    
    # Attempt 1: Standard JSON Parser
    try:
        parsed = json.loads(clean_text)
    except:
        pass
        
    # Attempt 2: Python AST Parser (Great for single quotes: {'key': 'val'})
    if not parsed:
        try:
            parsed = ast.literal_eval(clean_text)
        except:
            pass
            
    # Attempt 3: Regex Fallback (Crucial for truncated/broken JSON)
    if not parsed:
        parsed = {}
        
        # Regex for 'answer': Matches "answer": "1,000" OR "answer": 1000
        ans_match = re.search(r"['\"]answer['\"]\s*:\s*(?:['\"](.*?)['\"]|([0-9.,-]+))", clean_text, re.IGNORECASE)
        if ans_match:
            # Group 1 (quoted) or Group 2 (unquoted number)
            val = ans_match.group(1) if ans_match.group(1) is not None else ans_match.group(2)
            parsed['answer'] = val
            
        # Regex for 'confidence': Matches "confidence": 95
        conf_match = re.search(r"['\"]confidence['\"]\s*:\s*([0-9]+)", clean_text, re.IGNORECASE)
        if conf_match:
            parsed['confidence'] = int(conf_match.group(1))
            
        # Regex for 'reasoning': Captures everything until end of string or closing quote
        reas_match = re.search(r"['\"]reasoning['\"]\s*:\s*['\"](.*)", clean_text, re.IGNORECASE | re.DOTALL)
        if reas_match:
            raw_reas = reas_match.group(1)
            # Cleanup trailing JSON artifacts if they exist
            if raw_reas.endswith('"}') or raw_reas.endswith("'}n"):
                 raw_reas = raw_reas[:-2]
            elif raw_reas.endswith('"') or raw_reas.endswith("'"):
                 raw_reas = raw_reas[:-1]
            parsed['reasoning'] = raw_reas

    # Final Normalization
    answer = str(parsed.get('answer', '')).replace(',', '') # Remove commas from numbers
    confidence = parsed.get('confidence', 'N/A')
    reasoning = parsed.get('reasoning', '')

    return answer, confidence, reasoning
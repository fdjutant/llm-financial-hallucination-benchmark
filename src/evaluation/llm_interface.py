from pathlib import Path
from openai import OpenAI
import pandas as pd
import json
import os
import ast
import re
import random

# Set up API keys (for different providers)
openai_api_key = Path(Path(__file__).resolve().parents[2]/
                      "API_KEY"/"OPENAI_API_KEY").read_text().strip()
groq_api_key = Path(Path(__file__).resolve().parents[2]/
                   "API_KEY"/"GROQ_API_KEY").read_text().strip()
google_api_key = Path(Path(__file__).resolve().parents[2]/
                   "API_KEY"/"GOOGLE_API_KEY").read_text().strip()
nebius_api_key = Path(Path(__file__).resolve().parents[2]/
                   "API_KEY"/"NEBIUS_API_KEY").read_text().strip()
claude_api_key = Path(Path(__file__).resolve().parents[2]/
                   "API_KEY"/"CLAUDE_API_KEY").read_text().strip()   

class LLMOutput:
    def __init__(self, answer, confidence, reasoning, full_answer, provider):
        self.answer = answer
        self.confidence = confidence
        self.reasoning = reasoning
        self.full_answer = full_answer
        self.provider = provider

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
    
    elif model_name.startswith("deepseek-ai/") or model_name.startswith("Qwen/"):
        # Nebius models
        if not nebius_api_key:
            raise ValueError("NEBIUS_API_KEY not set.")
        return OpenAI(
            api_key=nebius_api_key,
            base_url="https://api.tokenfactory.nebius.com/v1/"
        ), "nebius"
    
    elif model_name.startswith("gemini-"):
        # Gemini models
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        return OpenAI(
            api_key=google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai"
        ), "gemini"
        
def call_llm_with_prompt(model, prompt):
    client, provider = get_client_for_model(model)
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
    return LLMOutput(answer, confidence, reasoning, llm_answer, provider)

def evaluate_with_xbrl_context(qa_pairs_path, 
                               evaluation_output_path, 
                               raw_output_path):

    df = pd.read_csv(qa_pairs_path)
    print(f"Loaded {len(df)} QA pairs")
    
    # Mix of OpenAI, Groq, and Gemini models
    models_to_test = [
        # OpenAI (paid)
        # "gpt-4o-mini",
        # "gpt-4o",
        
        # Groq (FREE - 14,400 requests/day)
        # "llama-3.3-70b-versatile",
        # "openai/gpt-oss-120b",

        # Financial specialist
        "deepseek-ai/DeepSeek-R1-0528",
        "Qwen/Qwen3-235B-A22B-Instruct-2507"
        
        # Gemini
        # "gemini-2.5-flash"
    ]
    
    results = []
    all_llm_outputs = []
    
    for idx, row in df.iterrows():
        id = row['id']
        question = row['generated_question']
        ground_truth_value = row['ground_truth_value']
        entity = row['entity_name']
        year = row['year']
        metric = row['original_metric']
        segment = row['segment']
        
        document = generate_document(segment, entity, year, metric, ground_truth_value)

        for model in models_to_test:
            try:
                # LAYER 1: COMPREHENSION EVALUATION
                prompt_rag = (
                f"You are a financial analyst reading an annual report excerpt.\n\n"
                f"FINANCIAL STATEMENT:\n"
                f"{document}\n"
                f"Based ONLY on the financial statement above, answer the question.\n"
                f"QUESTION: {question}\n"
                f"RESPONSE: {{'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}\n"
                f"CONFIDENCE SCALE:\n"
                f"90-100: Well-known figure. Example: 'Tesco's 2023 revenue was ~£57.7bn' (conf: 95)\n"
                f"70-89:  Confident in this knowledge. Example: 'Operating profit is calculated as gross profit minus operating expenses' (conf: 85)\n"
                f"50-69:  Reasonable inference but may be slightly off. Example: 'Barclays 2022 net interest income ~£13bn' (conf: 65)\n"
                f"30-49:  Educated guess. Example: 'Average FTSE 100 dividend yield is around 4%' (conf: 40)\n"
                f"0-29:   Very uncertain - if this low, respond UNKNOWN instead (conf: 15)\n\n"
                )             

                llm_output_rag = call_llm_with_prompt(model, prompt_rag)
                answer_rag = llm_output_rag.answer
                correct_rag = compare_answers(answer_rag, ground_truth_value)
                raw_output_rag = llm_output_rag.full_answer
                
                # LAYER 2: KNOWLEDGE EVALUATION
                prompt_knowledge = (
                f"You are a financial analyst.\n\n"
                f"QUESTION: {question}\n"
                f"RESPONSE: {{'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}\n"
                f"CONFIDENCE SCALE:\n"
                f"90-100: Well-known figure. Example: 'Tesco's 2023 revenue was ~£57.7bn' (conf: 95)\n"
                f"70-89:  Confident in this knowledge. Example: 'Operating profit is calculated as gross profit minus operating expenses' (conf: 85)\n"
                f"50-69:  Reasonable inference but may be slightly off. Example: 'Barclays 2022 net interest income ~£13bn' (conf: 65)\n"
                f"30-49:  Educated guess. Example: 'Average FTSE 100 dividend yield is around 4%' (conf: 40)\n"
                f"0-29:   Very uncertain - if this low, respond UNKNOWN instead (conf: 15)\n\n"
                )

                llm_output_knowledge = call_llm_with_prompt(model, prompt_knowledge)
                answer_knowledge = llm_output_knowledge.answer
                correct_knowledge = compare_answers(answer_knowledge, ground_truth_value)
                raw_output_knowledge = llm_output_knowledge.full_answer
                
                hallucinated = (  # Did it hallucinate? (Wrong + Confident, and not empty)
                    not correct_knowledge and 
                    answer_knowledge.strip() != "" and
                    "unknown" not in answer_knowledge.lower()
                )

                # LAYER 3: CONTRADICTION (ADVERSARIAL)
                fake_value = modify_value(ground_truth_value, noise=0.15)
                prompt_adversarial = (
                f"You are a financial analyst.\n\n"
                f"Source A: {metric} = {ground_truth_value}\n"
                f"Source B: {metric} = {fake_value}\n"
                f"Question: {question}\n"
                f"Which source is correct?\n"
                f"RESPONSE: {{'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}\n"
                f"CONFIDENCE SCALE:\n"
                f"90-100: Well-known figure. Example: 'Tesco's 2023 revenue was ~£57.7bn' (conf: 95)\n"
                f"70-89:  Confident in this knowledge. Example: 'Operating profit is calculated as gross profit minus operating expenses' (conf: 85)\n"
                f"50-69:  Reasonable inference but may be slightly off. Example: 'Barclays 2022 net interest income ~£13bn' (conf: 65)\n"
                f"30-49:  Educated guess. Example: 'Average FTSE 100 dividend yield is around 4%' (conf: 40)\n"
                f"0-29:   Very uncertain - if this low, respond UNKNOWN instead (conf: 15)\n\n"
                )

                llm_output_adversarial = call_llm_with_prompt(model, prompt_adversarial)
                answer_adversarial = llm_output_adversarial.answer
                trusted_correct_source = str(ground_truth_value) in str(answer_adversarial)
                raw_output_adversarial = llm_output_adversarial.full_answer

                results.append({
                    'id': id,
                    'model': model,
                    'question': question,
                    'ground_truth': ground_truth_value,
                    'fake_value': fake_value,
                    'answer_rag': answer_rag,
                    'answer_knowledge': answer_knowledge,
                    'answer_adversarial': answer_adversarial,
                    'rag_accuracy': correct_rag,
                    'knowledge_accuracy': correct_knowledge,
                    'hallucinated': hallucinated,
                    'adversarial_correct': trusted_correct_source
                })

                all_llm_outputs.append({
                    'id': id,
                    'model': model,
                    'question': question,
                    'ground_truth': ground_truth_value,
                    'fake_value': fake_value,
                    'raw_output_rag': raw_output_rag,
                    'raw_output_knowledge': raw_output_knowledge,
                    'raw_output_adversarial': raw_output_adversarial
                })
                
                print(f"✓ {model} - Row {idx}")
                
            except Exception as e:
                print(f"✗ {model} - Row {idx}: {e}")
                # Use safe defaults for failed outputs
                results.append({
                    'id': id,
                    'model': model,
                    'question': question,
                    'ground_truth': ground_truth_value,
                    'rag_accuracy': False,
                    'knowledge_accuracy': False,
                    'hallucinated': False,
                    'adversarial_correct': False
                })

                all_llm_outputs.append({
                    'id': id,
                    'model': model,
                    'question': question,
                    'ground_truth': ground_truth_value,
                    'raw_output_rag': '',
                    'raw_output_knowledge': '',
                    'raw_output_adversarial': ''
                })
        
        if (idx + 1) % 5 == 0:
            print(f"\n--- Evaluated {idx + 1}/{len(df)} QA pairs ---\n")
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(evaluation_output_path, index=False)

    raw_output_df = pd.DataFrame(all_llm_outputs)
    raw_output_df.to_csv(raw_output_path, index=False)
        
    return output_df, raw_output_df

def compare_answers(answer, ground_truth):
    """
    Compare the model's answer with the ground truth value.
    Returns True if they match (as strings, ignoring commas and whitespace), else False.
    """
    if answer is None or ground_truth is None:
        return False
    a = str(answer).replace(',', '').strip().lower()
    g = str(ground_truth).replace(',', '').strip().lower()
    return a == g

def modify_value(value, noise=0.1, min_noise=1.0):
    """
    Modifies a float value by a random percentage (default ±10%).
    If the value is zero, applies a minimum absolute noise (default 1.0).
    """
    try:
        num = float(value)
        if num == 0.0:
            delta = min_noise
        else:
            delta = abs(num) * noise
        new_value = num + random.uniform(-delta, delta)
        return str(round(new_value, 2))
    except Exception as e:
        print(f"Warning: Could not modify value '{value}'. Error: {e}. Returning original.")
        return str(value)

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

def generate_response(client, provider, model, prompt):
    """
    Generate a response from the client based on the given prompt.
    """
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=200,
        # response_format={"type": "json_object"} if provider == "openai" else None
    )

def generate_document(segment, entity, year, metric, ground_truth_value):
    """
    Generate a document string based on the segment type, entity, year, metric, and ground truth value.
    """
    if segment == "Income_Statement":
        return (
            f"CONSOLIDATED INCOME STATEMENT ({entity}, {year})\n"
            f"{metric}: {ground_truth_value}\n"
        )
    elif segment == "Balance_Sheet":
        return (
            f"CONSOLIDATED BALANCE SHEET ({entity}, {year})\n"
            f"{metric}: {ground_truth_value}\n"
        )
    elif segment == "Cash_Flow":
        return (
            f"CONSOLIDATED STATEMENT OF CASH FLOWS ({entity}, {year})\n"
            f"{metric}: {ground_truth_value}\n"
        )
    elif segment == "Company_Specific_Metric":
        return (
            f"CONSOLIDATED STATEMENT OF CASH FLOWS ({entity}, {year})\n"
            f"{metric}: {ground_truth_value}\n"
        )
    else:
        return f"{metric}: {ground_truth_value}\n"
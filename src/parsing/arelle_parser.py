from arelle import Cntlr, FileSource
from itertools import islice
from datetime import timedelta

def load_model_xbrl_old(entry_html_path: str):
    cntlr = Cntlr.Cntlr(logFileName=None)  # no GUI
    model_manager = cntlr.modelManager
    model_xbrl = model_manager.load(entry_html_path)
    return model_xbrl

def load_model_xbrl(filing_basefolder: str,
                    entry_html_path: str):
    print(f"Loading XBRL model from entry HTML: {entry_html_path}")
    print(f"Using filing base folder: {filing_basefolder}")
    cntlr = Cntlr.Cntlr(logFileName=None)
    file_source = FileSource.FileSource(filing_basefolder, cntlr)
    model_xbrl = cntlr.modelManager.load(entry_html_path,
                                         fileSource=file_source)
    return model_xbrl

def extract_context_rows(model_xbrl, filing_id: str):
    rows = []
    debug = True  # set True to only process a slice for debugging
    contexts_iter = islice(model_xbrl.contexts.items(), 10, 12) if debug else model_xbrl.contexts.items()
    for ctx_id, ctx in contexts_iter:
        entity = ctx.entityIdentifier  # (scheme, identifier)
        period = ctx.period
        row = {
            "filing_id": filing_id,
            "context_id": ctx_id,
            "entity_scheme": entity[0],
            "entity_identifier": entity[1],
            # "period_type": "instant" if period.isInstantPeriod else "duration",
            "instant": getattr(period, "instantDatetime", None),
            "period_start": getattr(period, "startDatetime", None),
            "period_end": getattr(period, "endDatetime", None),
            "dimensional_qualifier": ctx.qnameDims,  # you’ll JSON-ify this
        }
        rows.append(row)
    return rows

def extract_fact_rows(model_xbrl, filing_id: str):
    rows = []
    debug = True  # set True to only process the first fact for debugging
    facts_iter = islice(model_xbrl.facts, 10, 12) if debug else model_xbrl.facts
    for fact in facts_iter:
        if fact.isNil:
            continue

        start_date = getattr(fact.context, "startDatetime", None)
        end_date = getattr(fact.context, "endDatetime", None)

        row = {
            "filing_id": filing_id,
            "context_id": fact.contextID,
            "start_date": str(start_date.date()) if start_date else None,
            "end_date": str(end_date.date()-timedelta(days=1)) if end_date else None,
            "raw_name": str(fact.qname),              # prefix:localName
            "taxonomy_domain": fact.qname.prefix,     # prefix
            "data_type": "numeric" if fact.isNumeric else "string",
            "unit_ref": "GBP" if fact.unitID == "u-1" else fact.unitID,
            "decimals": getattr(fact, "decimals", None),
            "value_text": fact.textValue if not fact.isNumeric else None,
            "value_numeric": fact.xValue if fact.isNumeric else None,
        }
        rows.append(row)
    return rows

def safe_dict(d):
    """Convert all keys in a dict to str recursively and handle non-serializable objects."""
    if isinstance(d, dict):
        return {str(k): safe_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [safe_dict(i) for i in d]
    elif hasattr(d, '__str__') and not isinstance(d, (str, int, float, bool, type(None))):
        return str(d)
    else:
        return d

def debug_attributes(items, projectRoot, name="items"):
    """
    Debug and dump attributes for any iterable of objects (facts, contexts, etc.).
    Usage:
      debug_attributes(model_xbrl.facts, projectRoot, "fact")
      debug_attributes(model_xbrl.contexts.values(), projectRoot, "context")
    """
    attributes = _extract_all_attributes(items)
    if not attributes:
        print("No attributes found.")
        return
    print(f"{list(attributes[0].keys())}\n")
    output_file = (projectRoot / "data" / "processed" / "debug" / f"{name}_attributes_debug.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for key, value in attributes[0].items():
            f.write(f"{key}: {value}\n")
    print(f"Attributes written to {output_file}")

def debug_fact_attributes(model_xbrl, projectRoot):
    return debug_attributes(model_xbrl.facts, projectRoot, "fact")

def debug_context_attributes(model_xbrl, projectRoot):
    return debug_attributes(model_xbrl.contexts.values(), projectRoot, "context")

def _extract_all_attributes(items):
    """
    Extract values of all attributes from an iterable of objects.
    Returns a list of dicts (one per object).
    """
    extracted = []
    for obj in items:
        extracted.append({attr: getattr(obj, attr, None) for attr in dir(obj)})
    return extracted
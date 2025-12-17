from arelle import Cntlr, FileSource
from itertools import islice
from datetime import timedelta
import re

def load_model_xbrl(filing_basefolder: str, entry_html_path: str):
    
    cntlr = Cntlr.Cntlr(logFileName=None)
    file_source = FileSource.FileSource(filing_basefolder, cntlr)
    model_xbrl = cntlr.modelManager.load(entry_html_path,
                                         fileSource=file_source)

    return model_xbrl

def extract_context_rows(model_xbrl, filing_id: str):
    rows = []
    debug = False  # set True to only process a slice for debugging
    contexts_iter = islice(model_xbrl.contexts.items(), 50, 51) if debug else model_xbrl.contexts.items()
    for ctx_id, ctx in contexts_iter:
        entity = ctx.entityIdentifier  # (scheme, identifier)
        period = ctx.period
        row = {
            "filing_id": filing_id,
            "context_id": ctx_id,
            "entity_scheme": entity[0],
            "entity_identifier": entity[1],
            "period_type": "instant" if ctx.isInstantPeriod else "duration",
            "instant": str(getattr(ctx, "instantDatetime", None))[:10],
            "period_start": str(getattr(ctx, "startDatetime", None))[:10],
            "period_end": str(getattr(ctx, "endDatetime", None))[:10],
            "dimensional_qualifier": ctx.qnameDims,  # you’ll JSON-ify this
        }
        rows.append(row)
    return rows

def extract_fact_rows(model_xbrl, filing_id: str):
    rows = []
    debug = False  # set True to only process the first fact for debugging
    facts_iter = islice(model_xbrl.facts, 50, 51) if debug else model_xbrl.facts
    for fact in facts_iter:
        if fact.isNil:
            continue

        unit = getattr(fact, "unitID", None)
        unit_map = {
            "u-1": "GBP",
            "u-2": "GBP_per_Share",
            "Unit_GBP_per_Share": "GBP_per_Share",
        }
        unit_ref = None if unit is None else unit_map.get(unit, unit)

        row = {
            "filing_id": filing_id,
            "context_id": fact.contextID,
            "raw_name": str(fact.qname),              # prefix:localName
            "taxonomy_domain": fact.qname.prefix,     # prefix
            "data_type": "numeric" if fact.isNumeric else "string",
            "unit_ref": unit_ref,
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

def debug_fact_attributes(model_xbrl, projectRoot, object_number: int = 32):
    return _debug_attributes(model_xbrl.facts, projectRoot, "fact", object_number)

def debug_context_attributes(model_xbrl, projectRoot, object_number: int = 32):
    return _debug_attributes(model_xbrl.contexts.values(), projectRoot, "context", object_number)

def _debug_attributes(items, projectRoot, name="items", object_number: int = 32):
    """
    Debug and dump attributes for any iterable of objects (facts, contexts, etc.)
    Usage:
      debug_attributes(model_xbrl.facts, projectRoot, "fact", object_number=5)
      debug_attributes(model_xbrl.contexts.values(), projectRoot, "context")
    The object_number defaults to 32 but is clamped to the available range.
    """
    attributes = _extract_all_attributes(items)
    if not attributes:
        print("No attributes found")
        return
    # clamp the requested index into the valid range
    if object_number is None:
        idx = 0
    else:
        idx = max(0, min(object_number, len(attributes) - 1))

    output_file = (projectRoot / "data" / "processed" /
                   "debug" / f"{name}_attributes_debug.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        attrs = safe_dict(attributes[idx])
        for key in sorted(attrs.keys()):
            f.write(f"{key}: {attrs[key]}\n")
    print(f"Attributes written to {output_file} (object index {idx})")

def _extract_all_attributes(items):
    """
    Extract values of all attributes from an iterable of objects.
    Returns a list of dicts (one per object).
    """
    extracted = []
    for obj in items:
        extracted.append({attr: getattr(obj, attr, None) for attr in dir(obj)})
    return extracted
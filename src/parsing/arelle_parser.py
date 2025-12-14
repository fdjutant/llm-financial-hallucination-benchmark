from arelle import Cntlr

def load_model_xbrl(entry_html_path: str):
    cntlr = Cntlr.Cntlr(logFileName=None)  # no GUI
    model_manager = cntlr.modelManager
    model_xbrl = model_manager.load(entry_html_path)
    return model_xbrl

def extract_context_rows(model_xbrl, filing_id: str):
    rows = []
    for ctx_id, ctx in model_xbrl.contexts.items():
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
    for fact in model_xbrl.facts:
        if fact.isNil:
            continue
        row = {
            "filing_id": filing_id,
            "context_id": fact.contextID,
            "raw_name": str(fact.qname),              # prefix:localName
            "taxonomy_domain": fact.qname.prefix,     # prefix
            "data_type": "numeric" if fact.isNumeric else "string",
            "unit_ref": fact.unitID if fact.isNumeric else None,
            "decimals": getattr(fact, "decimals", None),
            "value_text": fact.textValue if not fact.isNumeric else None,
            "value_numeric": fact.xValue if fact.isNumeric else None,
        }
        rows.append(row)
    return rows

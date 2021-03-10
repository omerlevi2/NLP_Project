def get_evidence_ids(startqa_example):
    def get_evidence_strings(evidence):
        if isinstance(evidence, str):
            return [evidence]
        ids = []
        for evi in evidence:
            ids.extend(get_evidence_strings(evi))
        return ids

    evidence = startqa_example['evidence']
    evidence_ids = get_evidence_strings(evidence)
    return [x for x in evidence_ids if not x == 'operation' and not x == 'no_evidence']
"""Conservative corpus-title resolution shared by routing and retrieval."""
import re
import unicodedata


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def resolve_document_targets(query, documents):
    """Return unique exact title/ID matches and text with those mentions masked.

    No fuzzy entity inference: ambiguous title/year matches remain unresolved.
    Mask only the recognized entity and its adjacent catalog year, preserving
    independent 'as of 2020', 'latest', or other lifecycle requests.
    """
    text = normalize(query)
    candidates = {}
    for document in documents:
        title = normalize(document['title'])
        year = str(document.get('year') or '')
        aliases = {title}
        if re.fullmatch(r'(?:19|20)\d{2}', year):
            aliases.add(re.sub(r'^' + year + r'\s+', '', title))
        for alias in aliases:
            if len(alias.split()) < 3:
                continue
            for match in re.finditer(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text):
                start, end = match.span()
                prefix = re.search(r'\b((?:19|20)\d{2})\s+$', text[:start])
                # A conflicting edition year must not resolve to this catalog row.
                if prefix:
                    if prefix[1] != year:
                        continue
                    start = prefix.start()
                candidates.setdefault((start, end), set()).add(document['document_id'])
        for match in re.finditer(r'(?<!\w)' + re.escape(document['document_id'].casefold()) + r'(?!\w)', text):
            candidates.setdefault(match.span(), set()).add(document['document_id'])
    selected, spans = [], []
    for (start, end), ids in sorted(candidates.items(), key=lambda pair: -(pair[0][1]-pair[0][0])):
        if len(ids) != 1 or any(start < b and end > a for a, b in spans):
            continue
        spans.append((start, end))
        selected.extend(ids)
    chars = list(text)
    for start, end in spans:
        chars[start:end] = ' ' * (end-start)
    return list(dict.fromkeys(selected)), ''.join(chars)


def comparison_targets(query, documents):
    ids, residual = resolve_document_targets(query, documents)
    comparing = re.search(r'\b(?:compar\w*|differ\w*|versus|vs|between|contrast\w*)\b', residual)
    return ids if len(ids) > 1 and comparing else []

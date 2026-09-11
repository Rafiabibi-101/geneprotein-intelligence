import os
import re
import html
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st
from google import genai
from google.genai import types

APP_TITLE = "GeneProtein Intelligence"
APP_VERSION = "5.0"
DEFAULT_MODEL = "gemini-2.5-flash"

TIMEOUT = 25
UNIPROT_LIMIT = 30
PUBMED_LIMIT = 12
CLINVAR_LIMIT = 12
PDB_LIMIT = 12
ALPHAFOLD_LIMIT = 3

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA = "https://data.rcsb.org/rest/v1/core/entry"
ALPHAFOLD_APIS = ["https://alphafold.com/api/prediction", "https://alphafold.ebi.ac.uk/api/prediction"]
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
NCBI_TOOL = "GeneProteinIntelligence"

# A small set of high-value natural-language aliases. They are intentionally
# conservative: ambiguity is shown instead of silently guessing.
QUERY_HINTS = {
    "collagen type ii": "COL2A1",
    "collagen type 2": "COL2A1",
    "collagen ii": "COL2A1",
    "collagen 2": "COL2A1",
    "type ii collagen": "COL2A1",
    "type 2 collagen": "COL2A1",
    "beta globin": "HBB",
    "beta-globin": "HBB",
    "hemoglobin beta": "HBB",
    "haemoglobin beta": "HBB",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def config(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return value or os.getenv(name, default)


def safe_error(exc: Exception) -> str:
    text = clean(exc)
    return text[:260] if text else "Source unavailable."


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


def exactish(a: str, b: str) -> bool:
    return norm(a) == norm(b)


def aliases(value: Any) -> List[str]:
    if isinstance(value, list):
        vals = [clean(x) for x in value]
    else:
        vals = re.split(r"[,;|]", clean(value))
    return list(dict.fromkeys(x for x in vals if x))


def ncbi_params(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    p = {"tool": NCBI_TOOL, **(extra or {})}
    email = config("NCBI_EMAIL")
    key = config("NCBI_API_KEY")
    if email:
        p["email"] = email
    if key:
        p["api_key"] = key
    return p


def _request(method: str, url: str, **kwargs):
    headers = {
        "User-Agent": f"{NCBI_TOOL}/{APP_VERSION}",
        "Accept": "application/json, text/plain, */*",
    }
    headers.update(kwargs.pop("headers", {}))
    for attempt in range(4):
        try:
            r = requests.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < 3:
                    time.sleep(min(2 ** attempt, 8))
                    continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt >= 3:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Request failed.")


def get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return _request("GET", url, params=params or {}).json()


def get_text(url: str, params: Dict[str, Any] | None = None) -> str:
    return _request("GET", url, params=params or {}).text


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _request(
        "POST",
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
    ).json()


def first_ci(d: Dict[str, Any], *keys: str) -> Any:
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low:
            return low[k.lower()]
    return None


def uniprot_gene_names(record: Dict[str, Any]) -> Tuple[str, List[str]]:
    primary = ""
    names = []
    for g in record.get("genes", []) or []:
        if not isinstance(g, dict):
            continue
        primary_name = clean((g.get("geneName") or {}).get("value"))
        if primary_name and not primary:
            primary = primary_name
        if primary_name:
            names.append(primary_name)
        for s in g.get("synonyms", []) or []:
            v = clean((s or {}).get("value"))
            if v:
                names.append(v)
    return primary, list(dict.fromkeys(names))


def uniprot_protein_name(record: Dict[str, Any]) -> str:
    pd = record.get("proteinDescription") or {}
    rec = pd.get("recommendedName") or {}
    n = clean((rec.get("fullName") or {}).get("value"))
    if n:
        return n
    submitted = pd.get("submittedName") or []
    if submitted:
        n = clean(((submitted[0] or {}).get("fullName") or {}).get("value"))
    return n


def comment_text(record: Dict[str, Any], kind: str) -> List[str]:
    out = []
    for c in record.get("comments", []) or []:
        if c.get("commentType") != kind:
            continue
        for t in c.get("texts", []) or []:
            v = clean((t or {}).get("value"))
            if v:
                out.append(v)
    return list(dict.fromkeys(out))


def locations(record: Dict[str, Any]) -> List[str]:
    out = []
    for c in record.get("comments", []) or []:
        if c.get("commentType") != "SUBCELLULAR LOCATION":
            continue
        for loc in c.get("subcellularLocations", []) or []:
            v = clean(((loc or {}).get("location") or {}).get("value"))
            if v:
                out.append(v)
    return list(dict.fromkeys(out))


def pdb_crossrefs(record: Dict[str, Any]) -> List[str]:
    out = []
    for x in record.get("uniProtKBCrossReferences", []) or []:
        if x.get("database") == "PDB":
            pid = clean(x.get("id"))
            if pid:
                out.append(pid)
    return list(dict.fromkeys(out))


def score_uniprot(record: Dict[str, Any], query: str) -> int:
    q = clean(query)
    primary, names = uniprot_gene_names(record)
    pname = uniprot_protein_name(record)
    ql = q.lower()
    score = 0

    if exactish(primary, q):
        score += 220
    if any(exactish(x, q) for x in names):
        score += 170
    if exactish(pname, q):
        score += 150
    if ql and ql in pname.lower():
        score += 75
    if ql and pname.lower().startswith(ql):
        score += 25
    if record.get("entryType") == "UniProtKB reviewed (Swiss-Prot)":
        score += 35
    if clean((record.get("organism") or {}).get("scientificName")) == "Homo sapiens":
        score += 35
    return score


def parse_uniprot(record: Dict[str, Any]) -> Dict[str, Any]:
    primary, names = uniprot_gene_names(record)
    acc = clean(record.get("primaryAccession"))
    seq = record.get("sequence") or {}
    gene_ids = []
    for x in record.get("uniProtKBCrossReferences", []) or []:
        if x.get("database") == "GeneID":
            gene_ids.append(clean(x.get("id")))
    return {
        "accession": acc,
        "entry_name": clean(record.get("uniProtkbId")),
        "reviewed": record.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
        "protein_name": uniprot_protein_name(record),
        "gene_name": primary,
        "gene_aliases": names,
        "organism": clean((record.get("organism") or {}).get("scientificName")),
        "length": seq.get("length"),
        "mass": seq.get("molWeight"),
        "function": comment_text(record, "FUNCTION"),
        "localization": locations(record),
        "disease": comment_text(record, "DISEASE"),
        "ptm": comment_text(record, "PTM"),
        "gene_ids": list(dict.fromkeys(gene_ids)),
        "pdb_ids": pdb_crossrefs(record),
        "url": f"https://www.uniprot.org/uniprotkb/{acc}" if acc else "",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def search_uniprot(query: str) -> Dict[str, Any]:
    q = clean(query)
    if not q:
        return {}

    # Reviewed human first, then a broader human search if needed.
    queries = [
        f'(organism_id:9606) AND (reviewed:true) AND ("{q}")',
        f'(organism_id:9606) AND ("{q}")',
    ]
    records = []
    for uq in queries:
        data = get_json(UNIPROT_SEARCH, {
            "query": uq,
            "format": "json",
            "size": UNIPROT_LIMIT,
        })
        records = data.get("results", []) or []
        if records:
            break

    ranked = sorted(records, key=lambda r: score_uniprot(r, q), reverse=True)
    candidates = []
    for r in ranked[:12]:
        parsed = parse_uniprot(r)
        candidates.append({
            **parsed,
            "score": score_uniprot(r, q),
        })

    if not candidates:
        return {}

    best = candidates[0].copy()
    best["match_score"] = best.pop("score")
    best["candidate_count"] = len(candidates)
    best["candidates"] = candidates
    return best


@st.cache_data(ttl=3600, show_spinner=False)
def ncbi_gene_search(query: str) -> Dict[str, Any]:
    q = clean(query)
    if not q:
        return {}

    terms = [
        f'"{q}"[Gene Name] AND 9606[Taxonomy ID]',
        f'"{q}"[Gene Symbol] AND 9606[Taxonomy ID]',
        f'"{q}"[All Fields] AND 9606[Taxonomy ID]',
    ]
    ids = []
    for term in terms:
        data = get_json(EUTILS + "/esearch.fcgi", ncbi_params({
            "db": "gene", "term": term, "retmode": "json", "retmax": 12
        }))
        for x in data.get("esearchresult", {}).get("idlist", []) or []:
            if x not in ids:
                ids.append(x)
        if ids:
            break

    if not ids:
        return {}

    data = get_json(EUTILS + "/esummary.fcgi", ncbi_params({
        "db": "gene", "id": ",".join(ids[:12]), "retmode": "json"
    }))
    result = data.get("result", {}) or {}
    candidates = []

    for gid in ids:
        doc = result.get(gid, {})
        if not isinstance(doc, dict):
            continue
        symbol = clean(first_ci(doc, "Name", "NomenclatureSymbol", "Symbol"))
        other = aliases(first_ci(doc, "OtherAliases", "Synonyms", "Synonym"))
        desc = clean(first_ci(doc, "Summary", "Description", "DescriptionLong"))
        chrom = clean(first_ci(doc, "Chromosome", "chromosome"))
        loc = clean(first_ci(doc, "MapLocation", "Maplocation", "maplocation"))
        score = 0
        if exactish(symbol, q):
            score += 220
        if any(exactish(a, q) for a in other):
            score += 170
        if q.lower() in desc.lower() and q:
            score += 15
        candidates.append({
            "gene_id": clean(first_ci(doc, "uid", "GeneID", "GeneId") or gid),
            "symbol": symbol,
            "aliases": other,
            "description": desc,
            "chromosome": chrom,
            "map_location": loc,
            "score": score,
            "genomic_info": first_ci(doc, "GenomicInfo", "genomicinfo"),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if not candidates:
        return {}

    best = candidates[0].copy()
    best["url"] = f"https://www.ncbi.nlm.nih.gov/gene/{best['gene_id']}"
    best["candidates"] = candidates[:12]
    return best


def resolve_query_hint(query: str) -> Tuple[str, str]:
    q = clean(query)
    key = q.lower()
    if key in QUERY_HINTS:
        return QUERY_HINTS[key], "interpreted biological synonym"
    return q, ""


def resolve_entity(query: str) -> Dict[str, Any]:
    original = clean(query)
    lookup, hint = resolve_query_hint(original)
    failures = []

    gene = {}
    uni = {}
    try:
        gene = ncbi_gene_search(lookup)
    except Exception as e:
        failures.append(f"NCBI Gene: {safe_error(e)}")

    try:
        uni = search_uniprot(lookup)
    except Exception as e:
        failures.append(f"UniProt: {safe_error(e)}")

    gene_score = gene.get("score", 0)
    uni_score = uni.get("match_score", 0)

    # Exact gene symbols are safest.
    if gene and gene_score >= 200:
        symbol = gene.get("symbol") or lookup
        return {
            "type": "Gene / Protein",
            "query": original,
            "lookup_query": lookup,
            "symbol": symbol,
            "gene": gene,
            "uniprot": uni,
            "failures": failures,
            "hint": hint,
            "confidence": "High",
        }

    # Exact reviewed UniProt gene/protein matches are also safe.
    if uni and uni_score >= 200:
        symbol = uni.get("gene_name") or lookup
        # If several near-equal candidates exist, do not silently guess.
        near = [c for c in uni.get("candidates", []) if c.get("score", 0) >= uni_score - 25]
        if len(near) > 1 and not exactish(symbol, lookup):
            return {
                "type": "Ambiguous",
                "query": original,
                "lookup_query": lookup,
                "candidates": near[:8],
                "failures": failures,
                "hint": hint,
            }
        return {
            "type": "Gene / Protein",
            "query": original,
            "lookup_query": lookup,
            "symbol": symbol,
            "gene": gene,
            "uniprot": uni,
            "failures": failures,
            "hint": hint,
            "confidence": "High",
        }

    # Broad terms such as "collagen" should never silently become an arbitrary protein.
    combined = []
    for c in gene.get("candidates", [])[:8]:
        combined.append({
            "kind": "gene",
            "id": c.get("gene_id"),
            "symbol": c.get("symbol"),
            "label": c.get("description") or "NCBI Gene",
            "score": c.get("score", 0),
        })
    for c in uni.get("candidates", [])[:8]:
        combined.append({
            "kind": "protein",
            "id": c.get("accession"),
            "symbol": c.get("gene_name"),
            "label": c.get("protein_name") or "UniProt protein",
            "score": c.get("score", 0),
            "accession": c.get("accession"),
        })
    combined.sort(key=lambda x: x.get("score", 0), reverse=True)

    if len(combined) > 1 and combined[0].get("score", 0) >= 80:
        top = combined[:8]
        # Remove duplicate symbols where possible.
        seen = set()
        unique = []
        for c in top:
            k = (c.get("symbol") or "").upper()
            if k and k not in seen:
                unique.append(c)
                seen.add(k)
        if len(unique) > 1:
            return {
                "type": "Ambiguous",
                "query": original,
                "lookup_query": lookup,
                "candidates": unique,
                "failures": failures,
                "hint": hint,
            }

    chem = search_pubchem(original)
    if chem:
        return {
            "type": "Metabolite / Small molecule",
            "query": original,
            "lookup_query": lookup,
            "gene": {},
            "uniprot": {},
            "chem": chem,
            "failures": failures,
            "hint": hint,
        }

    if gene or uni:
        return {
            "type": "Low confidence",
            "query": original,
            "lookup_query": lookup,
            "gene": gene,
            "uniprot": uni,
            "failures": failures,
            "hint": hint,
        }

    return {
        "type": "Not resolved",
        "query": original,
        "lookup_query": lookup,
        "failures": failures,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed(gene_symbol: str, gene_aliases: List[str], free_query: str, limit: int = PUBMED_LIMIT) -> Dict[str, Any]:
    terms = []
    if gene_symbol:
        terms.append(f'"{gene_symbol}"[Title/Abstract]')
    for a in gene_aliases[:8]:
        if a and norm(a) != norm(gene_symbol):
            terms.append(f'"{a}"[Title/Abstract]')
    if free_query and norm(free_query) not in {norm(gene_symbol), *(norm(x) for x in gene_aliases)}:
        terms.append(f'"{free_query}"[Title/Abstract]')

    if not terms:
        return {"count": 0, "papers": [], "query": ""}

    # Do not require humans[MeSH Terms]; relevant papers may not carry that index term.
    term = "(" + " OR ".join(terms) + ")"
    search = get_json(EUTILS + "/esearch.fcgi", ncbi_params({
        "db": "pubmed", "term": term, "retmode": "json",
        "retmax": limit, "sort": "relevance",
    }))
    res = search.get("esearchresult", {}) or {}
    ids = res.get("idlist", []) or []
    count = int(res.get("count", "0") or 0)

    if not ids:
        return {"count": count, "papers": [], "query": term}

    xml = get_text(EUTILS + "/efetch.fcgi", ncbi_params({
        "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"
    }))
    return {"count": count, "papers": parse_pubmed_xml(xml), "query": term}


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_text)
    out = []
    for article in root.findall(".//PubmedArticle"):
        pmid = clean(article.findtext(".//PMID"))
        node = article.find(".//ArticleTitle")
        title = clean("".join(node.itertext())) if node is not None else ""
        abstract = []
        for n in article.findall(".//Abstract/AbstractText"):
            t = clean("".join(n.itertext()))
            label = clean(n.attrib.get("Label"))
            if t:
                abstract.append(f"{label}: {t}" if label else t)
        year = clean(article.findtext(".//PubDate/Year"))
        if not year:
            year = clean(article.findtext(".//PubDate/MedlineDate"))[:4]
        authors = []
        for a in article.findall(".//AuthorList/Author"):
            last = clean(a.findtext("LastName"))
            ini = clean(a.findtext("Initials"))
            if last:
                authors.append(f"{last} {ini}".strip())
        out.append({
            "pmid": pmid, "title": title, "abstract": " ".join(abstract),
            "journal": clean(article.findtext(".//Journal/Title")),
            "year": year, "authors": ", ".join(authors[:6]),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        })
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def search_clinvar(symbol: str, limit: int = CLINVAR_LIMIT) -> List[Dict[str, str]]:
    if not symbol:
        return []
    data = get_json(EUTILS + "/esearch.fcgi", ncbi_params({
        "db": "clinvar",
        "term": f"{symbol}[gene]",
        "retmode": "json", "retmax": limit, "sort": "relevance",
    }))
    ids = data.get("esearchresult", {}).get("idlist", []) or []
    if not ids:
        return []

    summary = get_json(EUTILS + "/esummary.fcgi", ncbi_params({
        "db": "clinvar", "id": ",".join(ids), "retmode": "json"
    }))
    result = summary.get("result", {}) or {}
    out = []
    for uid in ids:
        d = result.get(uid, {})
        if not isinstance(d, dict):
            continue
        acc = clean(first_ci(d, "accessionversion", "accession", "rcv_accession"))
        title = clean(first_ci(d, "title", "name", "variation_name"))
        sig = clean(first_ci(
            d, "clinical_significance", "clinicalsignificance",
            "clinical_significance_description"
        ))
        vid = clean(first_ci(d, "variationid", "variation_id", "uid") or uid)
        out.append({
            "uid": uid,
            "accession": acc,
            "title": title or f"ClinVar variation {vid}",
            "significance": sig,
            "variation_id": vid,
            "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{vid}/",
        })
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def pdb_details(pdb_ids: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out = []
    for pid in pdb_ids[:PDB_LIMIT]:
        try:
            d = get_json(f"{RCSB_DATA}/{pid}")
            info = d.get("rcsb_entry_info", {}) or {}
            entry = d.get("entry", {}) or {}
            struct = d.get("struct", {}) or {}
            methods = info.get("experimental_method", []) or []
            if isinstance(methods, str):
                methods = [methods]
            res = info.get("resolution_combined") or []
            if isinstance(res, (int, float)):
                res = [res]
            out.append({
                "pdb_id": pid,
                "title": clean(struct.get("title")) or f"PDB structure {pid}",
                "methods": [clean(x) for x in methods if clean(x)],
                "resolution": res[0] if res else None,
                "deposit_date": clean((entry.get("rcsb_accession_info") or {}).get("deposit_date")),
                "url": f"https://www.rcsb.org/structure/{pid}",
                "viewer": f"https://www.rcsb.org/3d-viewer/{pid}",
            })
        except Exception:
            continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def search_pdb(symbol: str, accession: str, known_ids: Tuple[str, ...]) -> List[Dict[str, Any]]:
    if known_ids:
        found = pdb_details(known_ids)
        if found:
            return found

    ids = []
    for value in [accession, symbol]:
        if not value:
            continue
        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": value},
            },
            "return_type": "entry",
            "request_options": {
                "pager": {"start": 0, "rows": PDB_LIMIT},
                "sort": [{"sort_by": "score", "direction": "desc"}],
            },
        }
        try:
            d = post_json(RCSB_SEARCH, payload)
            for item in d.get("result_set", []) or []:
                pid = clean(item.get("identifier"))
                if pid and pid not in ids:
                    ids.append(pid)
        except Exception:
            continue
        if len(ids) >= PDB_LIMIT:
            break

    return pdb_details(tuple(ids))



def _deep_find(obj: Any, keys: set[str]) -> Any:
    """Find a value by key in nested AlphaFold API payloads, tolerating schema changes."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in {x.lower() for x in keys}:
                return v
        for v in obj.values():
            found = _deep_find(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _deep_find(v, keys)
            if found is not None:
                return found
    return None


def _first_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list) and value:
        return _first_number(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def alphafold_predictions(accession: str) -> List[Dict[str, Any]]:
    """Retrieve current AlphaFold DB prediction metadata for a UniProt accession.

    The API changed during the 2026 transition, so the parser deliberately accepts
    both current and legacy field names and uses the current alphafold.com endpoint
    first, with the EBI endpoint as a fallback.
    """
    acc = clean(accession)
    if not acc:
        return []
    encoded = requests.utils.quote(acc, safe="")
    payload = None
    last_error = None
    for base in ALPHAFOLD_APIS:
        try:
            payload = get_json(f"{base}/{encoded}")
            break
        except Exception as e:
            last_error = e
    if payload is None:
        return []

    raw = payload if isinstance(payload, list) else payload.get("predictions", payload.get("results", []))
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for item in (raw or [])[:ALPHAFOLD_LIMIT]:
        if not isinstance(item, dict):
            continue
        entry_id = clean(_deep_find(item, {"entryId", "entry_id", "modelId", "model_id"}))
        version = _deep_find(item, {"latestVersion", "latest_version", "version", "modelVersion", "model_version"})
        start = _deep_find(item, {"uniprotStart", "uniprot_start", "start"})
        end = _deep_find(item, {"uniprotEnd", "uniprot_end", "end"})
        mean_plddt = _first_number(_deep_find(item, {"meanPlddt", "mean_plddt", "meanPlddtScore", "mean_confidence", "meanConfidence"}))
        max_pae = _first_number(_deep_find(item, {"maxPredictedAlignedError", "max_predicted_aligned_error", "maxPae", "max_pae"}))
        pdb_url = clean(_deep_find(item, {"pdbUrl", "pdb_url"}))
        cif_url = clean(_deep_find(item, {"cifUrl", "cif_url", "mmcifUrl", "mmcif_url"}))
        pae_image = clean(_deep_find(item, {"paeImageUrl", "pae_image_url"}))
        pae_doc = clean(_deep_find(item, {"paeDocUrl", "pae_doc_url"}))
        model_date = clean(_deep_find(item, {"modelCreatedDate", "model_created_date", "createdDate"}))
        if not entry_id:
            frag = f"F{1}"
            entry_id = f"AF-{acc}-{frag}"
        entry_url = f"https://alphafold.ebi.ac.uk/entry/{entry_id}"
        out.append({
            "entry_id": entry_id,
            "version": clean(version),
            "start": start,
            "end": end,
            "mean_plddt": mean_plddt,
            "max_pae": max_pae,
            "pdb_url": pdb_url,
            "cif_url": cif_url,
            "pae_image_url": pae_image,
            "pae_doc_url": pae_doc,
            "model_date": model_date,
            "entry_url": entry_url,
        })
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def search_pubchem(query: str) -> Dict[str, Any]:
    try:
        encoded = requests.utils.quote(clean(query), safe="")
        cid_data = get_json(f"{PUBCHEM}/compound/name/{encoded}/cids/JSON")
        cids = cid_data.get("IdentifierList", {}).get("CID", []) or []
        if not cids:
            return {}
        cid = cids[0]
        props = get_json(
            f"{PUBCHEM}/compound/cid/{cid}/property/"
            "Title,IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        )
        p = (props.get("PropertyTable", {}).get("Properties", [{}]) or [{}])[0]
        return {
            "cid": cid,
            "title": clean(p.get("Title")),
            "iupac": clean(p.get("IUPACName")),
            "formula": clean(p.get("MolecularFormula")),
            "weight": p.get("MolecularWeight"),
            "smiles": clean(p.get("CanonicalSMILES")),
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        }
    except Exception:
        return {}


def retrieve_evidence(entity: Dict[str, Any]) -> Dict[str, Any]:
    a = dict(entity)
    symbol = a.get("symbol") or ""
    g = a.get("gene") or {}
    u = a.get("uniprot") or {}
    failures = list(a.get("failures", []))

    try:
        a["clinvar"] = search_clinvar(symbol)
    except Exception as e:
        a["clinvar"] = []
        failures.append(f"ClinVar: {safe_error(e)}")

    try:
        a["pdb"] = search_pdb(
            symbol,
            u.get("accession", ""),
            tuple(u.get("pdb_ids", [])),
        )
    except Exception as e:
        a["pdb"] = []
        failures.append(f"RCSB PDB: {safe_error(e)}")

    try:
        a["alphafold"] = alphafold_predictions(u.get("accession", ""))
    except Exception as e:
        a["alphafold"] = []
        failures.append(f"AlphaFold DB: {safe_error(e)}")

    aliases_for_lit = list(dict.fromkeys(
        (g.get("aliases", []) or []) + (u.get("gene_aliases", []) or [])
    ))
    try:
        a["literature"] = search_pubmed(symbol, aliases_for_lit, a.get("query", ""))
    except Exception as e:
        a["literature"] = {"count": 0, "papers": [], "query": ""}
        failures.append(f"PubMed: {safe_error(e)}")

    a["failures"] = failures
    return a


def evidence_text(a: Dict[str, Any]) -> str:
    if a.get("type") == "Metabolite / Small molecule":
        c = a.get("chem", {})
        return f"""
SEARCH TERM: {a.get('query')}
ENTITY TYPE: Metabolite / Small molecule
SOURCE — PubChem
CID: {c.get('cid')}
Name: {c.get('title')}
IUPAC name: {c.get('iupac')}
Formula: {c.get('formula')}
Molecular weight: {c.get('weight')}
PubChem URL: {c.get('url')}
""".strip()

    g, u = a.get("gene", {}), a.get("uniprot", {})
    lines = [
        f"SEARCH TERM: {a.get('query')}",
        f"RESOLVED SYMBOL: {a.get('symbol')}",
        f"CONFIDENCE: {a.get('confidence', '')}",
        "",
        "SOURCE — NCBI Gene",
        f"Gene ID: {g.get('gene_id')}",
        f"Symbol: {g.get('symbol')}",
        f"Aliases: {', '.join(g.get('aliases', []))}",
        f"Description: {g.get('description')}",
        f"Chromosome: {g.get('chromosome')}",
        f"Map location: {g.get('map_location')}",
        f"URL: {g.get('url')}",
        "",
        "SOURCE — UniProt",
        f"Accession: {u.get('accession')}",
        f"Protein: {u.get('protein_name')}",
        f"Gene: {u.get('gene_name')}",
        f"Reviewed: {u.get('reviewed')}",
        f"Length: {u.get('length')}",
        f"Function: {' | '.join(u.get('function', []))}",
        f"Localization: {' | '.join(u.get('localization', []))}",
        f"Disease annotations: {' | '.join(u.get('disease', []))}",
        f"PDB cross-references: {', '.join(u.get('pdb_ids', []))}",
        f"URL: {u.get('url')}",
        "",
        "SOURCE — ClinVar",
    ]
    for x in a.get("clinvar", []):
        lines.append(
            f"{x.get('accession')} | {x.get('title')} | "
            f"{x.get('significance')} | variation {x.get('variation_id')}"
        )
    lines.append("")
    lines.append("SOURCE — RCSB PDB")
    for x in a.get("pdb", []):
        lines.append(
            f"{x.get('pdb_id')} | {x.get('title')} | "
            f"{', '.join(x.get('methods', []))} | resolution {x.get('resolution')}"
        )
    lines.append("")
    lines.append("SOURCE — AlphaFold Protein Structure Database")
    for x in a.get("alphafold", []):
        lines.append(
            f"{x.get('entry_id')} | version {x.get('version')} | "
            f"mean pLDDT {x.get('mean_plddt')} | max PAE {x.get('max_pae')} | "
            f"entry {x.get('entry_url')}"
        )
    lit = a.get("literature", {})
    lines += ["", "SOURCE — PubMed", f"Matching result count: {lit.get('count', 0)}"]
    for x in lit.get("papers", []):
        lines.append(
            f"PMID {x.get('pmid')} | {x.get('title')} | "
            f"{x.get('journal')} | {x.get('year')} | {x.get('abstract')}"
        )
    return "\n".join(lines)


def generate_ai(evidence: str) -> str:
    key = config("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing from Streamlit Secrets.")
    model = config("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=key)

    instruction = """
You are the evidence-synthesis layer of GeneProtein Intelligence, a research
and education application for biotechnology students.

Use ONLY the retrieved evidence. Never invent facts, identifiers, citations,
structures, variants, disease claims, or literature.
Do not diagnose or recommend treatment.
Do not claim the literature set is exhaustive.
Do not claim to have read full papers when only abstracts were retrieved.

Explain identifiers in plain scientific language. Clearly separate:
- established annotation,
- database-reported association,
- interpretation/inference,
- and missing evidence.

For ClinVar, explain that classifications represent submitted/aggregated
clinical interpretations and are not a patient-specific diagnosis.
For PDB, explain that a structure is an experimental molecular snapshot and
include the experimental method/resolution when supplied.
For AlphaFold DB, clearly label structures as computational predictions, not
experimental observations. Explain pLDDT as a model-confidence measure for
the predicted local 3D arrangement, not confidence that a biological claim is true.
Do not equate a high pLDDT score with clinical validity.

Produce:
1. What this entity is
2. Key biology
3. Gene/protein context
4. Disease and variant evidence
5. Structural evidence
6. Literature themes from the retrieved papers
7. Three useful research takeaways
8. Important limitations
9. Source IDs used
"""
    response = client.models.generate_content(
        model=model,
        contents=evidence,
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.15,
            max_output_tokens=3200,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text.")
    return text


def clear_analysis():
    for k in ["analysis", "report", "ai_error", "resolution"]:
        st.session_state.pop(k, None)


def current_query_is_analysis() -> bool:
    a = st.session_state.get("analysis")
    q = clean(st.session_state.get("gpi_query", ""))
    return bool(a and q and norm(a.get("query", "")) == norm(q))


def styles():
    st.markdown("""
    <style>
    .block-container {max-width:1500px;padding:1.3rem 2.2rem 3rem}
    .gpi-hero{padding:2.4rem;border:1px solid rgba(120,140,180,.22);
      border-radius:30px;margin-bottom:1rem;
      background:radial-gradient(circle at 88% 15%,rgba(70,170,185,.20),transparent 28%),
      radial-gradient(circle at 10% 100%,rgba(100,110,200,.15),transparent 34%),
      linear-gradient(135deg,rgba(30,45,75,.08),rgba(70,150,160,.06))}
    .gpi-kicker{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;font-weight:800;opacity:.62}
    .gpi-title{font-size:clamp(2.3rem,5vw,4.4rem);font-weight:900;letter-spacing:-.05em;line-height:.95;margin:.5rem 0 .8rem}
    .gpi-sub{font-size:1.02rem;line-height:1.6;opacity:.76;max-width:900px}
    .pill{display:inline-block;padding:.3rem .65rem;border-radius:999px;margin:.8rem .3rem 0 0;
      background:rgba(70,150,160,.10);border:1px solid rgba(70,150,160,.18);font-size:.75rem;font-weight:750}
    .card{padding:1.15rem 1.2rem;border:1px solid rgba(120,140,180,.20);border-radius:20px;
      background:rgba(128,128,128,.035);height:100%}
    .eyebrow{font-size:.72rem;text-transform:uppercase;letter-spacing:.11em;font-weight:800;opacity:.6}
    .big{font-size:1.45rem;font-weight:850;margin-top:.35rem}
    .muted{opacity:.68;font-size:.84rem}
    .section{padding:1rem 1.1rem;border-left:3px solid rgba(70,160,170,.58);
      background:rgba(100,130,160,.055);border-radius:0 15px 15px 0;margin:1rem 0}
    div[data-testid="stMetric"]{border:1px solid rgba(120,140,180,.20);padding:.75rem;border-radius:16px;background:rgba(120,140,180,.045)}
    .stTabs [data-baseweb="tab"]{border-radius:12px;padding:.55rem .75rem}
    .footer{text-align:center;opacity:.52;font-size:.78rem;padding-top:1.2rem}
    .science-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1.2rem 0 1.4rem}
    .science-tile{position:relative;overflow:hidden;min-height:170px;padding:1rem 1.1rem;border:1px solid rgba(120,140,180,.20);border-radius:22px;background:linear-gradient(145deg,rgba(120,140,180,.055),rgba(70,160,170,.07))}
    .science-tile svg{position:absolute;right:-4px;bottom:-12px;width:58%;height:78%;opacity:.76}
    .science-tile .tile-title{font-size:1.05rem;font-weight:850;position:relative;z-index:2}
    .science-tile .tile-copy{font-size:.78rem;opacity:.68;max-width:62%;line-height:1.45;position:relative;z-index:2;margin-top:.35rem}
    .hero-grid{display:grid;grid-template-columns:1.35fr .65fr;gap:20px;align-items:center}
    .hero-art{min-height:210px;display:flex;align-items:center;justify-content:center}
    .hero-art svg{width:100%;max-width:430px;height:auto}
    @media(max-width:850px){.science-strip{grid-template-columns:1fr}.hero-grid{grid-template-columns:1fr}.hero-art{display:none}}
    </style>
    """, unsafe_allow_html=True)


def render_gene(g):
    st.subheader("Genomic identity")
    if not g:
        st.info("No NCBI Gene record was retrieved for this resolved entity.")
        return
    c = st.columns(4)
    c[0].metric("NCBI Gene ID", g.get("gene_id") or "—")
    c[1].metric("Chromosome", g.get("chromosome") or "—")
    c[2].metric("Map location", g.get("map_location") or "—")
    c[3].metric("Symbol", g.get("symbol") or "—")
    st.markdown(
        f'<div class="section"><b>Description</b><br>{html.escape(g.get("description") or "Not available.")}</div>',
        unsafe_allow_html=True,
    )
    st.write("**Aliases:**", ", ".join(g.get("aliases", [])) or "Not available.")
    if g.get("url"):
        st.link_button("Open NCBI Gene record ↗", g["url"])


def render_protein(u):
    st.subheader("Protein identity")
    if not u:
        st.info("No UniProt protein record was retrieved.")
        return
    c = st.columns(4)
    c[0].metric("UniProt", u.get("accession") or "—")
    c[1].metric("Length", f"{u.get('length')} aa" if u.get("length") else "—")
    c[2].metric("Status", "Reviewed" if u.get("reviewed") else "Unreviewed")
    c[3].metric("PDB links", len(u.get("pdb_ids", [])))
    st.markdown(
        f'<div class="section"><b>Protein</b><br>{html.escape(u.get("protein_name") or "Not available.")}</div>',
        unsafe_allow_html=True,
    )
    x, y = st.columns(2)
    with x:
        st.write("**Function**")
        st.write("\n\n".join(u.get("function", [])) or "Not available.")
    with y:
        st.write("**Subcellular localization**")
        st.write(", ".join(u.get("localization", [])) or "Not available.")
    st.write("**Gene names / aliases**")
    st.write(", ".join(u.get("gene_aliases", [])) or "Not available.")
    if u.get("url"):
        st.link_button(f"Open UniProt {u.get('accession')} ↗", u["url"])


def render_clinvar(items):
    st.subheader("Clinical variant evidence")
    if not items:
        st.info("No ClinVar records were returned for this gene.")
        return
    st.caption("ClinVar is an archive of submitted/aggregated interpretations. It is not a patient-specific diagnosis.")
    for x in items:
        with st.expander(f"{x.get('accession') or 'ClinVar'} · {x.get('title') or 'Variation'}"):
            st.write("**Classification:**", x.get("significance") or "Not stated")
            st.write("**Variation ID:**", x.get("variation_id") or "—")
            st.link_button("Open original ClinVar record ↗", x["url"])


def render_pdb(items):
    st.subheader("Molecular structure")
    if not items:
        st.info("No matching experimental PDB structures were retrieved for this resolved protein.")
        return

    numeric = [x.get("resolution") for x in items if isinstance(x.get("resolution"), (int, float))]
    c = st.columns(3)
    c[0].metric("Structures", len(items))
    c[1].metric("Methods", len({m for x in items for m in x.get("methods", [])}))
    c[2].metric("Best resolution", f"{min(numeric):.2f} Å" if numeric else "—")

    st.markdown(
        '<div class="section"><b>Explore the molecule</b><br>'
        'Each entry below is an experimental PDB structure linked to the resolved protein. '
        'Use the interactive viewer for the molecular 3D representation.</div>',
        unsafe_allow_html=True,
    )

    for x in items:
        pid = x.get("pdb_id", "PDB")
        with st.expander(f"🧊 {pid} · {x.get('title') or 'Experimental structure'}"):
            a, b = st.columns([2, 1])
            with a:
                st.write("**Method:**", ", ".join(x.get("methods", [])) or "Not reported")
                st.write("**Resolution:**", f"{x.get('resolution')} Å" if x.get("resolution") else "Not reported")
                if x.get("deposit_date"):
                    st.write("**Deposited:**", x["deposit_date"])
            with b:
                st.link_button("Open RCSB structure ↗", x["url"])
                st.link_button("Open 3D viewer ↗", x["viewer"])
            # Embedded RCSB viewer. If embedding is blocked by a browser, the buttons above remain usable.
            try:
                import streamlit.components.v1 as components
                components.iframe(x["viewer"], height=430, scrolling=False)
            except Exception:
                pass


def render_alphafold(items: List[Dict[str, Any]], accession: str):
    st.subheader("🤖 Predicted structure — AlphaFold DB")
    if not accession:
        st.info("A UniProt accession is required to retrieve an AlphaFold prediction.")
        return
    if not items:
        st.info("No current AlphaFold DB prediction was returned for this UniProt accession.")
        st.caption("Not every protein or fragment is necessarily available in the current database.")
        return

    st.markdown(
        '<div class="section"><b>Experimental vs predicted</b><br>'
        'RCSB PDB structures are experimental observations. AlphaFold structures are '
        'AI-based predictions. Use the confidence information to judge model reliability; '
        'a prediction is not experimental evidence.</div>', unsafe_allow_html=True
    )
    for x in items:
        with st.expander(f"🧠 {x.get('entry_id') or 'AlphaFold model'} · predicted model", expanded=True):
            c = st.columns(4)
            c[0].metric("Mean pLDDT", f"{x['mean_plddt']:.1f}" if isinstance(x.get('mean_plddt'), (int, float)) else "—")
            c[1].metric("Max PAE", f"{x['max_pae']:.1f} Å" if isinstance(x.get('max_pae'), (int, float)) else "—")
            c[2].metric("Model version", x.get("version") or "—")
            covered = "—"
            if x.get("start") is not None and x.get("end") is not None:
                covered = f"{x.get('start')}–{x.get('end')}"
            c[3].metric("Residues", covered)

            if x.get("pae_image_url"):
                st.image(x["pae_image_url"], caption="Predicted Aligned Error (PAE)")
            elif x.get("pae_doc_url"):
                st.info("A raw PAE file is available from AlphaFold DB.")

            buttons = st.columns(4)
            buttons[0].link_button("Open AlphaFold model ↗", x.get("entry_url") or f"https://alphafold.ebi.ac.uk/entry/{accession}")
            if x.get("pdb_url"):
                buttons[1].link_button("Download PDB ↗", x["pdb_url"])
            if x.get("cif_url"):
                buttons[2].link_button("Download CIF ↗", x["cif_url"])
            if x.get("pae_doc_url"):
                buttons[3].link_button("Download PAE ↗", x["pae_doc_url"])

            try:
                import streamlit.components.v1 as components
                components.iframe(x.get("entry_url") or f"https://alphafold.ebi.ac.uk/entry/{accession}", height=520, scrolling=False)
            except Exception:
                pass

    st.caption("AlphaFold DB data are provided for research use with attribution. Verify important structural conclusions against primary sources and experimental evidence.")


def render_literature(lit):
    st.subheader("Scientific literature")
    count = lit.get("count", 0)
    papers = lit.get("papers", [])
    c1, c2 = st.columns(2)
    c1.metric("Matching PubMed records", f"{count:,}")
    c2.metric("Retrieved for this screen", len(papers))
    st.caption("The count is the database search result count. The displayed papers are a practical top set, not an exhaustive review.")
    for p in papers:
        with st.expander(f"{p.get('year') or 'Year'} · {p.get('title') or 'Untitled'}"):
            st.write("**PMID:**", p.get("pmid") or "—")
            st.write("**Journal:**", p.get("journal") or "—")
            if p.get("authors"):
                st.write("**Authors:**", p["authors"])
            st.write(p.get("abstract") or "Abstract unavailable.")
            if p.get("url"):
                st.link_button("Read on PubMed ↗", p["url"])


def render_chem(c):
    st.info("This query resolved to a small molecule rather than a human gene/protein.")
    cols = st.columns(4)
    cols[0].metric("PubChem CID", c.get("cid") or "—")
    cols[1].metric("Formula", c.get("formula") or "—")
    cols[2].metric("Molecular weight", c.get("weight") or "—")
    cols[3].metric("Name", c.get("title") or "—")
    st.write("**IUPAC name:**", c.get("iupac") or "—")
    if c.get("url"):
        st.link_button("Open PubChem record ↗", c["url"])


def render_ambiguity(a):
    st.warning(
        f'“{a.get("query")}” matches multiple biological entities. '
        "GPI will not silently guess."
    )
    if a.get("hint"):
        st.caption(f"Interpretation used: {a['hint']} → {a.get('lookup_query')}")
    candidates = a.get("candidates", [])
    labels = []
    mapping = {}
    for i, c in enumerate(candidates):
        symbol = c.get("symbol") or "Unknown"
        label = c.get("label") or c.get("kind", "candidate")
        accession = c.get("accession") or c.get("id") or ""
        text = f"{symbol} — {label}" + (f" [{accession}]" if accession else "")
        labels.append(text)
        mapping[text] = c

    if not labels:
        st.info("No clear candidates were returned. Try an official gene symbol or UniProt accession.")
        return

    choice = st.selectbox("Choose the entity you mean", labels, key="gpi_candidate_choice")
    if st.button("Use this entity", type="primary", key="gpi_use_candidate"):
        selected = mapping[choice]
        target = selected.get("symbol") or selected.get("accession")
        st.session_state["gpi_query"] = target
        st.session_state.pop("resolution", None)
        st.session_state.pop("analysis", None)
        st.rerun()


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
    styles()

    st.markdown("""
    <div class="gpi-hero">
      <div class="hero-grid">
        <div>
          <div class="gpi-kicker">Biomedical evidence workspace · v5.0</div>
          <div class="gpi-title">GeneProtein<br>Intelligence</div>
          <div class="gpi-sub">
            Resolve a gene, protein, or biological term first. Then connect it to
            curated protein annotation, genomic context, clinical-variant evidence,
            experimental and predicted structures, scientific literature, and an AI research brief.
          </div>
          <span class="pill">NCBI Gene</span><span class="pill">UniProt</span>
          <span class="pill">ClinVar</span><span class="pill">RCSB PDB</span>
          <span class="pill">PubMed</span><span class="pill">PubChem</span><span class="pill">AlphaFold DB</span>
        </div>
        <div class="hero-art" aria-hidden="true">
          <svg viewBox="0 0 500 260" xmlns="http://www.w3.org/2000/svg">
            <defs><linearGradient id="gpiA" x1="0" x2="1"><stop offset="0"/><stop offset="1" stop-opacity=".35"/></linearGradient></defs>
            <path d="M70 25 C150 75 150 185 70 235 M120 25 C40 75 40 185 120 235" fill="none" stroke="currentColor" stroke-width="9" stroke-linecap="round" opacity=".42"/>
            <g stroke="currentColor" stroke-width="5" opacity=".30"><path d="M65 55h60"/><path d="M51 85h83"/><path d="M48 115h86"/><path d="M48 145h86"/><path d="M51 175h83"/><path d="M65 205h60"/></g>
            <path d="M230 150 C190 100 255 55 300 92 C344 128 312 178 270 166 C225 153 230 210 285 220 C340 230 390 183 360 137 C332 95 385 52 430 78" fill="none" stroke="currentColor" stroke-width="13" stroke-linecap="round" opacity=".34"/>
            <path d="M235 151 C198 111 254 70 295 99 C326 121 310 155 278 151 C247 147 243 181 274 194 C310 209 356 177 339 144" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" opacity=".65"/>
            <path d="M425 35 v190" stroke="currentColor" stroke-width="12" stroke-linecap="round" opacity=".18"/>
            <path d="M405 50 h40 M405 80 h40 M405 110 h40 M405 140 h40 M405 170 h40 M405 200 h40" stroke="currentColor" stroke-width="5" opacity=".48"/>
          </svg>
        </div>
      </div>
    </div>
    <div class="science-strip">
      <div class="science-tile">
        <div class="tile-title">🧬 Gene</div><div class="tile-copy">Chromosome, locus, aliases and NCBI Gene context.</div>
        <svg viewBox="0 0 220 170" xmlns="http://www.w3.org/2000/svg"><path d="M55 8 C145 55 145 115 55 162 M95 8 C5 55 5 115 95 162" fill="none" stroke="currentColor" stroke-width="10" opacity=".42"/><g stroke="currentColor" stroke-width="4" opacity=".3"><path d="M49 34h52"/><path d="M34 60h67"/><path d="M32 86h69"/><path d="M34 112h67"/><path d="M49 138h52"/></g></svg>
      </div>
      <div class="science-tile">
        <div class="tile-title">🧪 Protein</div><div class="tile-copy">Function, localization, sequence length and UniProt evidence.</div>
        <svg viewBox="0 0 220 170" xmlns="http://www.w3.org/2000/svg"><path d="M30 120 C10 75 70 42 105 75 C135 103 110 137 78 124 C48 112 45 150 82 155 C126 161 182 126 160 84 C143 52 177 31 202 48" fill="none" stroke="currentColor" stroke-width="12" stroke-linecap="round" opacity=".4"/><circle cx="105" cy="75" r="11" fill="currentColor" opacity=".25"/><circle cx="160" cy="84" r="9" fill="currentColor" opacity=".25"/></svg>
      </div>
      <div class="science-tile">
        <div class="tile-title">🧊 Chromosome & structure</div><div class="tile-copy">Genomic location plus PDB experimental structures and AlphaFold predictions.</div>
        <svg viewBox="0 0 220 170" xmlns="http://www.w3.org/2000/svg"><path d="M155 15v140M125 15v140" stroke="currentColor" stroke-width="15" opacity=".16" stroke-linecap="round"/><g stroke="currentColor" stroke-width="5" opacity=".5"><path d="M125 35h30M125 65h30M125 95h30M125 125h30"/></g><path d="M35 115 C10 75 55 43 90 70 C120 94 95 128 67 118 C39 108 43 145 74 150 C111 155 135 128 119 96" fill="none" stroke="currentColor" stroke-width="10" opacity=".35" stroke-linecap="round"/></svg>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🧬 GPI")
        st.caption("Resolve → Verify → Explore → Synthesize")
        st.divider()
        st.write("**Quick searches**")
        for ex in ["EGFR", "BRCA1", "TP53", "HBB", "COL2A1", "collagen 2", "hemoglobin", "creatinine"]:
            if st.button(ex, use_container_width=True, key=f"quick_{ex}"):
                st.session_state["gpi_query"] = ex
                clear_analysis()
                st.rerun()
        st.divider()
        st.caption("For research and education. Not a diagnostic or treatment system.")
        if st.button("Clear", use_container_width=True):
            clear_analysis()
            st.rerun()

    query = st.text_input(
        "Search a gene, protein, or biological term",
        key="gpi_query",
        placeholder="Try EGFR, HBB, COL2A1, collagen 2, hemoglobin…",
        max_chars=120,
    )

    if st.session_state.get("last_rendered_query") != query:
        if st.session_state.get("last_rendered_query") is not None:
            clear_analysis()
        st.session_state["last_rendered_query"] = query

    if st.button("🔎 Analyze", type="primary", use_container_width=True):
        q = clean(query)
        clear_analysis()
        if len(q) < 2:
            st.error("Enter at least 2 characters.")
        else:
            with st.status(f"Resolving “{q}”…", expanded=True) as status:
                try:
                    resolution = resolve_entity(q)
                    st.session_state["resolution"] = resolution
                    status.update(label="Entity resolution complete", state="complete")
                except Exception as e:
                    st.session_state["resolution"] = {
                        "type": "Not resolved",
                        "query": q,
                        "failures": [safe_error(e)],
                    }
                    status.update(label="Resolution failed", state="error")
            st.rerun()

    resolution = st.session_state.get("resolution")
    if not resolution:
        st.markdown("""
        <div class="card">
          <div class="eyebrow">Ready</div>
          <div class="big">Search a biological entity to begin.</div>
          <div class="muted">GPI resolves the identity before it retrieves downstream evidence.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if norm(resolution.get("query", "")) != norm(query):
        clear_analysis()
        return

    rtype = resolution.get("type")
    if rtype == "Ambiguous":
        render_ambiguity(resolution)
        return

    if rtype == "Not resolved":
        st.error(f'GPI could not confidently resolve “{query}”.')
        if resolution.get("failures"):
            with st.expander("Technical details"):
                for f in resolution["failures"]:
                    st.write("• " + f)
        return

    if rtype == "Low confidence":
        st.warning(
            "GPI found evidence, but not a high-confidence identity. "
            "It will not present this as a definitive biological match."
        )
        return

    if rtype == "Metabolite / Small molecule":
        st.markdown(f"## {html.escape(query)}")
        render_chem(resolution.get("chem", {}))
        return

    # Retrieve downstream evidence only after identity is resolved.
    if st.session_state.get("analysis") is None:
        with st.status(f"Gathering evidence for {resolution.get('symbol')}…", expanded=True) as status:
            try:
                analysis = retrieve_evidence(resolution)
                st.session_state["analysis"] = analysis
                status.update(label="Evidence retrieved", state="complete")
            except Exception as e:
                st.session_state["analysis"] = {**resolution, "failures": resolution.get("failures", []) + [safe_error(e)]}
                status.update(label="Evidence retrieval incomplete", state="error")

    analysis = st.session_state.get("analysis")
    if not analysis or not current_query_is_analysis():
        return

    if "report" not in st.session_state and not st.session_state.get("ai_error"):
        with st.spinner("Gemini is synthesizing the retrieved evidence…"):
            try:
                st.session_state["report"] = generate_ai(evidence_text(analysis))
                st.session_state["ai_error"] = ""
            except Exception as e:
                st.session_state["ai_error"] = safe_error(e)

    g = analysis.get("gene", {})
    u = analysis.get("uniprot", {})
    cv = analysis.get("clinvar", [])
    pdb = analysis.get("pdb", [])
    af = analysis.get("alphafold", [])
    lit = analysis.get("literature", {"count": 0, "papers": []})
    symbol = analysis.get("symbol") or query

    st.markdown(f"""
    <div class="card">
      <div class="eyebrow">High-confidence resolved entity</div>
      <div class="big">{html.escape(symbol)}</div>
      <div class="muted">{html.escape(u.get("protein_name") or g.get("description") or "Biological entity")}</div>
    </div>
    """, unsafe_allow_html=True)

    if analysis.get("hint"):
        st.caption(f"Query interpretation: {analysis['hint']} → {analysis.get('lookup_query')}")

    cols = st.columns(5)
    cols[0].metric("Gene ID", g.get("gene_id") or "—")
    cols[1].metric("UniProt", u.get("accession") or "—")
    cols[2].metric("ClinVar records", len(cv))
    cols[3].metric("PDB structures", len(pdb))
    cols[4].metric("AlphaFold models", len(af))

    tabs = st.tabs([
        "Overview", "🧬 Gene", "🧪 Protein", "⚕️ Variants",
        "🧊 3D Structure", "📚 Literature", "🤖 AI Research", "🔗 Sources"
    ])

    with tabs[0]:
        st.subheader("Research snapshot")
        x, y = st.columns(2)
        with x:
            st.markdown(
                f'<div class="card"><div class="eyebrow">Genomic context</div>'
                f'<div class="big">{html.escape(g.get("symbol") or symbol)}</div>'
                f'<div class="muted">NCBI Gene {html.escape(g.get("gene_id") or "—")} · '
                f'Chromosome {html.escape(g.get("chromosome") or "—")} · '
                f'{html.escape(g.get("map_location") or "map location unavailable")}</div></div>',
                unsafe_allow_html=True)
        with y:
            st.markdown(
                f'<div class="card"><div class="eyebrow">Protein context</div>'
                f'<div class="big">{html.escape(u.get("accession") or "—")}</div>'
                f'<div class="muted">{html.escape(u.get("protein_name") or "Protein record unavailable")} · '
                f'{u.get("length") or "—"} aa · '
                f'{html.escape(", ".join(u.get("localization", [])) or "localization unavailable")}</div></div>',
                unsafe_allow_html=True)

        function_text = (u.get("function") or [g.get("description") or "No concise function annotation was retrieved."])[0]
        st.markdown(
            f'<div class="section"><b>Primary biological annotation</b><br>{html.escape(function_text)}</div>',
            unsafe_allow_html=True)

        if pdb or af:
            st.success(f"🧊 Structural evidence available: {len(pdb)} experimental PDB structure(s) and {len(af)} AlphaFold prediction(s). Open the 3D Structure tab.")
        if analysis.get("failures"):
            with st.expander("Source warnings"):
                for f in analysis["failures"]:
                    st.write("• " + f)

    with tabs[1]:
        render_gene(g)
    with tabs[2]:
        render_protein(u)
    with tabs[3]:
        render_clinvar(cv)
        if u.get("disease"):
            st.subheader("UniProt disease annotations")
            for d in u["disease"]:
                st.write("• " + d)
    with tabs[4]:
        render_pdb(pdb)
        st.divider()
        render_alphafold(af, u.get("accession", ""))
    with tabs[5]:
        render_literature(lit)
    with tabs[6]:
        st.subheader("AI Research Intelligence")
        st.caption("Gemini synthesizes only the evidence retrieved for this search.")
        if st.session_state.get("ai_error"):
            st.warning(st.session_state["ai_error"])
        elif st.session_state.get("report"):
            st.markdown(st.session_state["report"])
    with tabs[7]:
        st.subheader("Evidence sources")
        links = []
        if u.get("url"):
            links.append(("UniProt " + str(u.get("accession")), u["url"]))
        if g.get("url"):
            links.append(("NCBI Gene " + str(g.get("gene_id")), g["url"]))
        for x in cv:
            if x.get("url"):
                links.append(("ClinVar " + str(x.get("accession") or x.get("variation_id")), x["url"]))
        for x in pdb:
            if x.get("url"):
                links.append(("RCSB PDB " + str(x.get("pdb_id")), x["url"]))
        for x in af:
            if x.get("entry_url"):
                links.append(("AlphaFold " + str(x.get("entry_id")), x["entry_url"]))
        for x in lit.get("papers", []):
            if x.get("url"):
                links.append(("PubMed PMID " + str(x.get("pmid")), x["url"]))
        for label, url in links:
            st.link_button(label + " ↗", url)

    st.markdown(
        '<div class="footer">GPI 4.1 · Live evidence from public biomedical resources · '
        'Always verify important scientific claims against original records.</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


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
APP_VERSION = "3.0"
DEFAULT_MODEL = "gemini-2.5-flash"
TIMEOUT = 25
UNIPROT_LIMIT = 25
PUBMED_LIMIT = 12
CLINVAR_LIMIT = 12
PDB_LIMIT = 12

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA = "https://data.rcsb.org/rest/v1/core/entry"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
NCBI_TOOL = "GeneProteinIntelligence"


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


def ncbi_params(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    p = {"tool": NCBI_TOOL, **(extra or {})}
    email = config("NCBI_EMAIL")
    key = config("NCBI_API_KEY")
    if email:
        p["email"] = email
    if key:
        p["api_key"] = key
    return p


def get_json(url: str, params: Dict[str, Any] | None = None, retries: int = 2) -> Dict[str, Any]:
    headers = {"User-Agent": f"{NCBI_TOOL}/{APP_VERSION}", "Accept": "application/json, text/plain, */*"}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params or {}, headers=headers, timeout=TIMEOUT)
            if r.status_code == 429 and attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt >= retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Request failed.")


def get_text(url: str, params: Dict[str, Any] | None = None, retries: int = 2) -> str:
    headers = {"User-Agent": f"{NCBI_TOOL}/{APP_VERSION}"}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params or {}, headers=headers, timeout=TIMEOUT)
            if r.status_code == 429 and attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if attempt >= retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Request failed.")


def post_json(url: str, payload: Dict[str, Any], retries: int = 2) -> Dict[str, Any]:
    headers = {"User-Agent": f"{NCBI_TOOL}/{APP_VERSION}", "Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
            if r.status_code == 429 and attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt >= retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Request failed.")


def first_ci(d: Dict[str, Any], *keys: str) -> Any:
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low:
            return low[k.lower()]
    return None


def aliases(value: Any) -> List[str]:
    if isinstance(value, list):
        vals = [clean(x) for x in value]
    else:
        vals = re.split(r"[,;|]", clean(value))
    return list(dict.fromkeys(x for x in vals if x))


def exactish(a: str, b: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", a.lower()) == re.sub(r"[^a-z0-9]", "", b.lower())


def uniprot_gene_names(record: Dict[str, Any]) -> Tuple[str, List[str]]:
    primary = ""
    all_names = []
    for g in record.get("genes", []):
        n = clean(g.get("geneName", {}).get("value"))
        if n and not primary:
            primary = n
        if n:
            all_names.append(n)
        for s in g.get("synonyms", []):
            v = clean(s.get("value"))
            if v:
                all_names.append(v)
    return primary, list(dict.fromkeys(all_names))


def uniprot_protein_name(record: Dict[str, Any]) -> str:
    pd = record.get("proteinDescription", {})
    n = clean(pd.get("recommendedName", {}).get("fullName", {}).get("value"))
    if n:
        return n
    submitted = pd.get("submittedName", [])
    if submitted:
        return clean(submitted[0].get("fullName", {}).get("value"))
    return ""


def comment_text(record: Dict[str, Any], kind: str) -> List[str]:
    out = []
    for c in record.get("comments", []):
        if c.get("commentType") != kind:
            continue
        for t in c.get("texts", []):
            v = clean(t.get("value"))
            if v:
                out.append(v)
    return list(dict.fromkeys(out))


def locations(record: Dict[str, Any]) -> List[str]:
    out = []
    for c in record.get("comments", []):
        if c.get("commentType") != "SUBCELLULAR LOCATION":
            continue
        for loc in c.get("subcellularLocations", []):
            v = clean(loc.get("location", {}).get("value"))
            if v:
                out.append(v)
    return list(dict.fromkeys(out))


def pdb_crossrefs(record: Dict[str, Any]) -> List[str]:
    out = []
    for x in record.get("uniProtKBCrossReferences", []):
        if x.get("database") == "PDB":
            pid = clean(x.get("id"))
            if pid:
                out.append(pid)
    return list(dict.fromkeys(out))


def score_uniprot(record: Dict[str, Any], query: str) -> int:
    primary, names = uniprot_gene_names(record)
    pname = uniprot_protein_name(record)
    q = clean(query)
    score = 0
    if exactish(primary, q):
        score += 120
    if any(exactish(x, q) for x in names):
        score += 90
    if exactish(pname, q):
        score += 85
    if q.lower() in pname.lower() and q:
        score += 80
    if pname.lower().startswith(q.lower()) and q:
        score += 20
    if record.get("entryType") == "UniProtKB reviewed (Swiss-Prot)":
        score += 25
    if clean(record.get("organism", {}).get("scientificName")) == "Homo sapiens":
        score += 25
    return score


def parse_uniprot(record: Dict[str, Any]) -> Dict[str, Any]:
    primary, names = uniprot_gene_names(record)
    acc = clean(record.get("primaryAccession"))
    seq = record.get("sequence", {})
    gene_ids = []
    for x in record.get("uniProtKBCrossReferences", []):
        if x.get("database") == "GeneID":
            gene_ids.append(clean(x.get("id")))
    return {
        "accession": acc,
        "entry_name": clean(record.get("uniProtkbId") or record.get("uniProtKBCrossReferences", [{}])[0].get("id")),
        "reviewed": record.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
        "protein_name": uniprot_protein_name(record),
        "gene_name": primary,
        "gene_aliases": names,
        "organism": clean(record.get("organism", {}).get("scientificName")),
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


def search_uniprot(query: str) -> Dict[str, Any]:
    q = clean(query)
    # Broad human search, but explicitly rank exact gene/protein matches.
    params = {
        "query": f"({q}) AND organism_id:9606 AND reviewed:true",
        "format": "json",
        "size": UNIPROT_LIMIT,
    }
    data = get_json(UNIPROT_SEARCH, params)
    records = data.get("results", [])
    if not records:
        return {}

    ranked = sorted(records, key=lambda r: score_uniprot(r, q), reverse=True)
    best = ranked[0]
    result = parse_uniprot(best)
    result["match_score"] = score_uniprot(best, q)
    result["candidate_count"] = len(records)
    result["candidates"] = [
        {
            "accession": clean(r.get("primaryAccession")),
            "gene": uniprot_gene_names(r)[0],
            "protein": uniprot_protein_name(r),
            "reviewed": r.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
            "score": score_uniprot(r, q),
        }
        for r in ranked[:8]
    ]
    return result


def ncbi_gene_search(query: str) -> Dict[str, Any]:
    q = clean(query)
    terms = [
        f'"{q}"[Gene Name] AND 9606[Taxonomy ID]',
        f'"{q}"[Gene Symbol] AND 9606[Taxonomy ID]',
        f'"{q}"[All Fields] AND 9606[Taxonomy ID]',
    ]
    ids = []
    for term in terms:
        data = get_json(EUTILS + "/esearch.fcgi", ncbi_params({
            "db": "gene", "term": term, "retmode": "json", "retmax": 8
        }))
        for x in data.get("esearchresult", {}).get("idlist", []):
            if x not in ids:
                ids.append(x)
        if ids:
            break
    if not ids:
        return {}

    data = get_json(EUTILS + "/esummary.fcgi", ncbi_params({
        "db": "gene", "id": ",".join(ids[:8]), "retmode": "json"
    }))
    result = data.get("result", {})
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
            score += 120
        if any(exactish(a, q) for a in other):
            score += 90
        if q.lower() in desc.lower():
            score += 10
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
    if not candidates:
        return {}
    best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
    best["url"] = f"https://www.ncbi.nlm.nih.gov/gene/{best['gene_id']}"
    best["candidates"] = candidates[:8]
    return best


def search_pubmed(gene_symbol: str, gene_aliases: List[str], free_query: str, limit: int = PUBMED_LIMIT) -> Dict[str, Any]:
    terms = [f'"{gene_symbol}"[Title/Abstract]']
    for a in gene_aliases[:6]:
        if a and a.lower() != gene_symbol.lower():
            terms.append(f'"{a}"[Title/Abstract]')
    if free_query and free_query.lower() not in {gene_symbol.lower(), *(x.lower() for x in gene_aliases)}:
        terms.append(f'"{free_query}"[Title/Abstract]')
    term = "(" + " OR ".join(terms) + ") AND humans[MeSH Terms]"
    search = get_json(EUTILS + "/esearch.fcgi", ncbi_params({
        "db": "pubmed", "term": term, "retmode": "json", "retmax": limit, "sort": "relevance"
    }))
    res = search.get("esearchresult", {})
    ids = res.get("idlist", [])
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


def search_clinvar(symbol: str, limit: int = CLINVAR_LIMIT) -> List[Dict[str, str]]:
    if not symbol:
        return []
    data = get_json(EUTILS + "/esearch.fcgi", ncbi_params({
        "db": "clinvar",
        "term": f"{symbol}[gene] AND single_gene[prop]",
        "retmode": "json",
        "retmax": limit,
        "sort": "relevance",
    }))
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    summary = get_json(EUTILS + "/esummary.fcgi", ncbi_params({
        "db": "clinvar", "id": ",".join(ids), "retmode": "json"
    }))
    result = summary.get("result", {})
    out = []
    for uid in ids:
        d = result.get(uid, {})
        if not isinstance(d, dict):
            continue
        acc = clean(first_ci(d, "accessionversion", "accession", "rcv_accession"))
        title = clean(first_ci(d, "title", "name", "variation_name"))
        sig = clean(first_ci(d, "clinical_significance", "clinicalsignificance", "clinical_significance_description"))
        vid = clean(first_ci(d, "variationid", "variation_id", "uid") or uid)
        out.append({
            "uid": uid, "accession": acc, "title": title or f"ClinVar variation {vid}",
            "significance": sig, "variation_id": vid,
            "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{vid}/",
        })
    return out


def pdb_details(pdb_ids: List[str]) -> List[Dict[str, Any]]:
    out = []
    for pid in pdb_ids[:PDB_LIMIT]:
        try:
            d = get_json(f"{RCSB_DATA}/{pid}")
            info = d.get("rcsb_entry_info", {})
            entry = d.get("entry", {})
            struct = d.get("struct", {})
            methods = info.get("experimental_method", [])
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
                "deposit_date": clean(entry.get("rcsb_accession_info", {}).get("deposit_date")),
                "url": f"https://www.rcsb.org/structure/{pid}",
            })
        except Exception:
            continue
    return out


def search_pdb(symbol: str, accession: str, known_ids: List[str]) -> List[Dict[str, Any]]:
    # First use UniProt cross-references: this is the most precise path.
    if known_ids:
        found = pdb_details(known_ids)
        if found:
            return found

    identifiers = [x for x in [accession, symbol] if x]
    ids = []
    for text in identifiers:
        payload = {
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": text}},
            "return_type": "entry",
            "request_options": {"pager": {"start": 0, "rows": PDB_LIMIT}},
        }
        try:
            d = post_json(RCSB_SEARCH, payload)
            for item in d.get("result_set", []):
                pid = clean(item.get("identifier"))
                if pid and pid not in ids:
                    ids.append(pid)
        except Exception:
            continue
        if len(ids) >= PDB_LIMIT:
            break
    return pdb_details(ids)


def search_pubchem(query: str) -> Dict[str, Any]:
    # Used only as a fallback when no convincing human gene/protein entity resolves.
    try:
        encoded = requests.utils.quote(clean(query), safe="")
        cid_data = get_json(f"{PUBCHEM}/compound/name/{encoded}/cids/JSON")
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {}
        cid = cids[0]
        props = get_json(
            f"{PUBCHEM}/compound/cid/{cid}/property/"
            "Title,IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        )
        p = props.get("PropertyTable", {}).get("Properties", [{}])[0]
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


def resolve_entity(query: str) -> Dict[str, Any]:
    q = clean(query)
    gene = {}
    uni = {}
    failures = []
    try:
        gene = ncbi_gene_search(q)
    except Exception as e:
        failures.append(f"NCBI Gene: {e}")
    try:
        uni = search_uniprot(q)
    except Exception as e:
        failures.append(f"UniProt: {e}")

    # A convincing exact gene match wins. Otherwise a convincing UniProt match wins.
    gene_score = gene.get("score", 0)
    uni_score = uni.get("match_score", 0)

    if gene and gene_score >= 100:
        entity_type = "Gene / Protein"
        symbol = gene.get("symbol") or uni.get("gene_name") or q
    elif uni and uni_score >= 100:
        entity_type = "Protein"
        symbol = uni.get("gene_name") or q
    else:
        # Try PubChem for non-gene/non-protein terms such as creatinine.
        chem = search_pubchem(q)
        if chem:
            return {
                "type": "Metabolite / Small molecule",
                "query": q,
                "gene": {},
                "uniprot": {},
                "chem": chem,
                "failures": failures,
            }
        # Keep the best evidence even when the match is weak.
        if gene or uni:
            entity_type = "Possible gene/protein"
            symbol = gene.get("symbol") or uni.get("gene_name") or q
        else:
            return {"type": "Not resolved", "query": q, "failures": failures}

    return {
        "type": entity_type,
        "query": q,
        "symbol": symbol,
        "gene": gene,
        "uniprot": uni,
        "failures": failures,
    }


def generate_ai(evidence: str) -> str:
    key = config("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing from Streamlit Secrets.")
    model = config("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=key)
    instruction = """
You are the scientific synthesis layer of GeneProtein Intelligence.

Use ONLY the retrieved evidence below. Never invent facts or citations.
Do not diagnose, recommend treatment, or give patient-specific advice.
Do not claim the literature set is exhaustive.
Do not claim to have read full papers when only abstracts were retrieved.

Write for a biotechnology student who may be new to databases.
Translate identifiers into plain scientific language.
For ClinVar, explain what each classification means in database terms and
make clear that a database classification is not a patient diagnosis.
For PDB, explain what an experimental structure represents.
If evidence conflicts or is absent, say so explicitly.

Produce:
1. What this entity is
2. Key biology
3. Gene/protein context
4. Disease and variant evidence
5. Structural evidence
6. What the retrieved literature is mainly about
7. Three useful research takeaways
8. Limitations
9. Source list with database names and IDs
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


def evidence_text(a: Dict[str, Any]) -> str:
    if a["type"] == "Metabolite / Small molecule":
        c = a["chem"]
        return f"""
SEARCH TERM: {a['query']}
ENTITY TYPE: Metabolite / Small molecule
SOURCE — PubChem
CID: {c.get('cid')}
Name: {c.get('title')}
IUPAC name: {c.get('iupac')}
Formula: {c.get('formula')}
Molecular weight: {c.get('weight')}
PubChem URL: {c.get('url')}
""".strip()

    g = a.get("gene", {})
    u = a.get("uniprot", {})
    cv = a.get("clinvar", [])
    pdb = a.get("pdb", [])
    lit = a.get("literature", {})
    lines = [
        f"SEARCH TERM: {a['query']}",
        f"RESOLVED SYMBOL: {a.get('symbol')}",
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
    for x in cv:
        lines += [f"{x.get('accession')} | {x.get('title')} | {x.get('significance')} | variation {x.get('variation_id')}"]
    lines += ["", "SOURCE — RCSB PDB"]
    for x in pdb:
        lines += [f"{x.get('pdb_id')} | {x.get('title')} | {', '.join(x.get('methods', []))} | resolution {x.get('resolution')}"]
    lines += ["", "SOURCE — PubMed", f"Total relevant search results reported by NCBI: {lit.get('count', 0)}"]
    for x in lit.get("papers", []):
        lines += [f"PMID {x.get('pmid')} | {x.get('title')} | {x.get('journal')} | {x.get('year')} | {x.get('abstract')}"]
    return "\n".join(lines)


def clear_current():
    st.session_state.pop("analysis", None)
    st.session_state.pop("report", None)
    st.session_state.pop("ai_error", None)


def query_changed():
    # Prevent an old EGFR result from surviving while the user has typed a new query.
    current = st.session_state.get("gpi_query", "").strip()
    previous = st.session_state.get("gpi_active_query", "").strip()
    if previous and current.lower() != previous.lower():
        clear_current()


def run(query: str):
    clear_current()
    st.session_state["gpi_active_query"] = query

    with st.status(f"Resolving “{query}”…", expanded=True) as status:
        entity = resolve_entity(query)
        if entity["type"] == "Not resolved":
            status.update(label="No confident entity match", state="error")
            st.session_state["analysis"] = entity
            return

        if entity["type"] == "Metabolite / Small molecule":
            status.update(label="Small molecule identified", state="complete")
            st.session_state["analysis"] = entity
            return

        symbol = entity.get("symbol") or query
        g = entity.get("gene", {})
        u = entity.get("uniprot", {})

        st.write(f"Resolved entity: {symbol}")
        st.write("Retrieving ClinVar evidence…")
        try:
            entity["clinvar"] = search_clinvar(symbol)
        except Exception as e:
            entity["clinvar"] = []
            entity.setdefault("failures", []).append(f"ClinVar: {e}")

        st.write("Retrieving exact PDB structures…")
        try:
            entity["pdb"] = search_pdb(symbol, u.get("accession", ""), u.get("pdb_ids", []))
        except Exception as e:
            entity["pdb"] = []
            entity.setdefault("failures", []).append(f"RCSB PDB: {e}")

        st.write("Searching PubMed literature…")
        try:
            entity["literature"] = search_pubmed(
                symbol,
                g.get("aliases", []) + u.get("gene_aliases", []),
                query,
            )
        except Exception as e:
            entity["literature"] = {"count": 0, "papers": [], "query": ""}
            entity.setdefault("failures", []).append(f"PubMed: {e}")

        st.session_state["analysis"] = entity
        status.update(label="Evidence retrieval complete", state="complete")

    with st.spinner("Gemini is synthesizing the retrieved evidence…"):
        try:
            st.session_state["report"] = generate_ai(evidence_text(entity))
            st.session_state["ai_error"] = ""
        except Exception as e:
            st.session_state["report"] = ""
            st.session_state["ai_error"] = str(e)



def styles():
    st.markdown("""
    <style>
    :root {
      --gpi-border: rgba(120,140,180,.20);
      --gpi-soft: rgba(120,140,180,.07);
      --gpi-muted: rgba(120,120,130,.72);
    }
    .block-container {max-width:1500px;padding:1.2rem 2.2rem 3rem}
    header[data-testid="stHeader"] {background:transparent}
    .gpi-hero {
      position:relative; overflow:hidden; padding:2.2rem 2.3rem;
      border:1px solid var(--gpi-border); border-radius:30px;
      background:
        radial-gradient(circle at 88% 15%, rgba(80,170,190,.18), transparent 28%),
        radial-gradient(circle at 15% 110%, rgba(90,110,190,.13), transparent 35%),
        linear-gradient(135deg, rgba(30,45,75,.10), rgba(70,150,160,.06));
      margin-bottom:1.1rem;
    }
    .gpi-kicker {font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;font-weight:800;opacity:.62}
    .gpi-title {font-size:clamp(2.25rem,5vw,4.2rem);font-weight:900;letter-spacing:-.045em;line-height:.98;margin:.45rem 0 .8rem}
    .gpi-sub {font-size:1.03rem;line-height:1.6;opacity:.76;max-width:880px}
    .gpi-pill {
      display:inline-block;padding:.32rem .65rem;border-radius:999px;
      background:rgba(70,150,160,.10);border:1px solid rgba(70,150,160,.18);
      font-size:.76rem;font-weight:700;margin:.8rem .35rem 0 0
    }
    .gpi-search {
      padding:1.15rem;border:1px solid var(--gpi-border);border-radius:22px;
      background:var(--gpi-soft);margin-bottom:1.2rem
    }
    .gpi-card {
      padding:1.15rem 1.2rem;border:1px solid var(--gpi-border);
      border-radius:20px;background:rgba(128,128,128,.035);height:100%
    }
    .gpi-card-title {font-size:.76rem;text-transform:uppercase;letter-spacing:.10em;font-weight:800;opacity:.62}
    .gpi-big {font-size:1.45rem;font-weight:850;margin-top:.35rem}
    .gpi-small {font-size:.82rem;opacity:.66}
    .gpi-section {
      padding:1rem 1.1rem;border-left:3px solid rgba(80,160,170,.55);
      background:var(--gpi-soft);border-radius:0 15px 15px 0;margin:1rem 0
    }
    div[data-testid="stMetric"] {border:1px solid var(--gpi-border);padding:.8rem;border-radius:16px;background:var(--gpi-soft)}
    div[data-testid="stMetricLabel"] {font-size:.74rem}
    button[kind="primary"] {border-radius:14px;font-weight:800}
    .stTabs [data-baseweb="tab-list"] {gap:5px}
    .stTabs [data-baseweb="tab"] {border-radius:12px;padding:.55rem .8rem}
    .gpi-footer {text-align:center;opacity:.55;font-size:.78rem;padding-top:1rem}
    </style>
    """, unsafe_allow_html=True)


def render_gene(g: Dict[str, Any]):
    st.markdown("### Genomic identity")
    if not g:
        st.info("No confident NCBI Gene record was retrieved.")
        return
    c = st.columns(4)
    c[0].metric("NCBI Gene ID", g.get("gene_id") or "—")
    c[1].metric("Chromosome", g.get("chromosome") or "—")
    c[2].metric("Map location", g.get("map_location") or "—")
    c[3].metric("Official symbol", g.get("symbol") or "—")
    st.markdown('<div class="gpi-section"><b>Description</b><br>' +
                html.escape(g.get("description") or "Not available.") + '</div>',
                unsafe_allow_html=True)
    st.markdown("**Aliases**")
    st.write(", ".join(g.get("aliases", [])) or "Not available.")
    if g.get("url"):
        st.link_button("Open NCBI Gene record ↗", g["url"])


def render_protein(u: Dict[str, Any]):
    st.markdown("### Protein identity")
    if not u:
        st.info("No confident human UniProt record was retrieved.")
        return
    c = st.columns(4)
    c[0].metric("UniProt", u.get("accession") or "—")
    c[1].metric("Length", f"{u.get('length')} aa" if u.get("length") else "—")
    c[2].metric("Reviewed", "Swiss-Prot" if u.get("reviewed") else "—")
    c[3].metric("Structures", len(u.get("pdb_ids", [])))
    st.markdown(f'<div class="gpi-section"><b>Protein</b><br>{html.escape(u.get("protein_name") or "—")}</div>',
                unsafe_allow_html=True)
    x, y = st.columns(2)
    with x:
        st.markdown("**Function**")
        st.write("\n\n".join(u.get("function", [])) or "Not available.")
    with y:
        st.markdown("**Subcellular localization**")
        st.write(", ".join(u.get("localization", [])) or "Not available.")
    st.markdown("**Gene names / aliases**")
    st.write(", ".join(u.get("gene_aliases", [])) or "Not available.")
    if u.get("url"):
        st.link_button(f"Open UniProt {u.get('accession')} ↗", u["url"])


def render_clinvar(items):
    st.markdown("### Clinical variant evidence")
    if not items:
        st.info("No ClinVar records were returned for the resolved gene.")
        return
    st.caption("ClinVar is an archive of submitted evidence and interpretations; it is not a patient-specific diagnosis.")
    for x in items:
        title = x.get("title") or "ClinVar record"
        with st.expander(f"{x.get('accession') or 'ClinVar'}  ·  {title}"):
            st.write(f"**Classification:** {x.get('significance') or 'Not stated in retrieved summary.'}")
            st.write(f"**Variation ID:** {x.get('variation_id') or '—'}")
            if x.get("url"):
                st.link_button("Open original ClinVar record ↗", x["url"])


def render_pdb(items):
    st.markdown("### Molecular structure")
    if not items:
        st.info("No matching experimental PDB structures were retrieved for this resolved protein.")
        return
    st.markdown(
        '<div class="gpi-section"><b>Why this matters</b><br>'
        'PDB structures provide experimental snapshots of molecular architecture. '
        'Open a structure to explore the protein in three dimensions.</div>',
        unsafe_allow_html=True
    )
    cols = st.columns(4)
    cols[0].metric("Experimental structures", len(items))
    methods = sorted({m for x in items for m in x.get("methods", [])})
    cols[1].metric("Methods", len(methods))
    cols[2].metric("Best resolution", 
                   f"{min([x.get('resolution') for x in items if isinstance(x.get('resolution'), (int,float))]):.2f} Å"
                   if any(isinstance(x.get('resolution'), (int,float)) for x in items) else "—")
    cols[3].metric("PDB IDs", "Available")
    for x in items:
        pid = x.get("pdb_id", "PDB")
        with st.expander(f"🧊 {pid}  ·  {x.get('title') or 'Experimental structure'}"):
            a,b = st.columns(2)
            with a:
                st.write(f"**Method:** {', '.join(x.get('methods', [])) or 'Not reported'}")
                st.write(f"**Resolution:** {x.get('resolution')} Å" if x.get("resolution") else "**Resolution:** Not reported")
            with b:
                if x.get("deposit_date"):
                    st.write(f"**Deposited:** {x['deposit_date']}")
                st.link_button(f"Explore {pid} in RCSB PDB ↗", x["url"])


def render_literature(lit):
    st.markdown("### Scientific literature")
    count = lit.get("count", 0)
    papers = lit.get("papers", [])
    c1,c2 = st.columns(2)
    c1.metric("PubMed matches", f"{count:,}")
    c2.metric("Retrieved for review", len(papers))
    st.caption("The number above is the matching PubMed result count; the cards below are a practical top set, not an exhaustive literature review.")
    for p in papers:
        with st.expander(f"{p.get('year') or 'Year'}  ·  {p.get('title') or 'Untitled'}"):
            st.write(f"**PMID:** {p.get('pmid') or '—'}")
            st.write(f"**Journal:** {p.get('journal') or '—'}")
            if p.get("authors"):
                st.write(f"**Authors:** {p['authors']}")
            st.write(p.get("abstract") or "Abstract unavailable.")
            if p.get("url"):
                st.link_button("Read on PubMed ↗", p["url"])


def render_chem(c):
    st.markdown("### Small-molecule profile")
    st.info("This term resolved to a small molecule rather than a human gene/protein.")
    cols = st.columns(4)
    cols[0].metric("PubChem CID", c.get("cid") or "—")
    cols[1].metric("Formula", c.get("formula") or "—")
    cols[2].metric("Molecular weight", c.get("weight") or "—")
    cols[3].metric("Name", c.get("title") or "—")
    st.markdown(f"**IUPAC name:** {c.get('iupac') or '—'}")
    if c.get("url"):
        st.link_button("Open PubChem record ↗", c["url"])


def main():
    st.set_page_config(page_title="GeneProtein Intelligence", page_icon="🧬", layout="wide")
    styles()

    st.markdown("""
    <div class="gpi-hero">
      <div class="gpi-kicker">AI-powered biomedical research workspace</div>
      <div class="gpi-title">GeneProtein<br>Intelligence</div>
      <div class="gpi-sub">
        Resolve a biological entity, connect trusted biomedical evidence, explore molecular
        structure, discover literature, and generate a source-grounded research brief.
      </div>
      <span class="gpi-pill">NCBI Gene</span>
      <span class="gpi-pill">UniProt</span>
      <span class="gpi-pill">ClinVar</span>
      <span class="gpi-pill">RCSB PDB</span>
      <span class="gpi-pill">PubMed</span>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🧬 GPI")
        st.caption("Gene → Protein → Structure → Evidence → Insight")
        st.divider()
        st.markdown("**Try a search**")
        examples = ["EGFR", "BRCA1", "TP53", "HBB", "hemoglobin", "creatinine"]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"example_{ex}"):
                st.session_state["gpi_query"] = ex
                clear_current()
                st.session_state["gpi_active_query"] = ""
                st.rerun()
        st.divider()
        st.caption("Research & education only. Not a diagnostic or treatment system.")
        if st.button("Clear current analysis", use_container_width=True):
            clear_current()
            st.session_state["gpi_active_query"] = ""
            st.rerun()

    st.markdown('<div class="gpi-search">', unsafe_allow_html=True)
    query = st.text_input(
        "Search a gene, protein, or biological term",
        key="gpi_query",
        placeholder="Try EGFR, BRCA1, TP53, HBB, hemoglobin…",
        max_chars=120,
        on_change=query_changed,
    )
    c1,c2 = st.columns([5,1])
    with c1:
        st.caption("GPI resolves the entity before retrieving evidence, so results stay tied to the current search.")
    with c2:
        analyze = st.button("🔎 Analyze", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        q = query.strip()
        if not q:
            st.error("Enter a gene, protein, or biological term.")
        elif len(q) < 2:
            st.error("Please enter at least 2 characters.")
        else:
            run(q)

    a = st.session_state.get("analysis")
    if not a:
        st.markdown("""
        <div class="gpi-card">
          <div class="gpi-card-title">Ready for discovery</div>
          <div class="gpi-big">Search a gene or protein to begin.</div>
          <div class="gpi-small">GPI will resolve the entity and assemble evidence from multiple biomedical resources.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if a.get("type") == "Not resolved":
        st.error(f"GPI could not confidently resolve “{a.get('query')}”.")
        st.info("Try a gene symbol, protein name, alias, or common biological term.")
        return

    if a.get("type") == "Metabolite / Small molecule":
        st.markdown(f"## {a.get('query')}")
        render_chem(a.get("chem", {}))
        return

    g, u = a.get("gene", {}), a.get("uniprot", {})
    cv, pdb = a.get("clinvar", []), a.get("pdb", [])
    lit = a.get("literature", {"count": 0, "papers": []})
    symbol = a.get("symbol") or a.get("query")

    # Identity banner
    st.markdown(f"""
    <div class="gpi-card">
      <div class="gpi-card-title">Resolved biological entity</div>
      <div class="gpi-big">{html.escape(symbol)}</div>
      <div class="gpi-small">{html.escape(u.get("protein_name") or g.get("description") or a.get("type") or "Human biological entity")}</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    cols[0].metric("Gene ID", g.get("gene_id") or "—")
    cols[1].metric("UniProt", u.get("accession") or "—")
    cols[2].metric("Variants", len(cv))
    cols[3].metric("Structures", len(pdb))
    cols[4].metric("PubMed", f"{lit.get('count', 0):,}")

    tabs = st.tabs(["Overview", "🧬 Gene", "🧪 Protein", "⚕️ Variants", "🧊 3D Structure", "📚 Literature", "🤖 AI Research", "🔗 Sources"])

    with tabs[0]:
        st.markdown("### Research snapshot")
        x, y = st.columns(2)
        with x:
            st.markdown(
                f'<div class="gpi-card"><div class="gpi-card-title">Genomic context</div>'
                f'<div class="gpi-big">{html.escape(g.get("symbol") or symbol)}</div>'
                f'<div class="gpi-small">NCBI Gene {html.escape(g.get("gene_id") or "—")} · '
                f'Chromosome {html.escape(g.get("chromosome") or "—")} · '
                f'{html.escape(g.get("map_location") or "map location unavailable")}</div></div>',
                unsafe_allow_html=True)
        with y:
            st.markdown(
                f'<div class="gpi-card"><div class="gpi-card-title">Protein context</div>'
                f'<div class="gpi-big">{html.escape(u.get("accession") or "—")}</div>'
                f'<div class="gpi-small">{html.escape(u.get("protein_name") or "Protein record unavailable")} · '
                f'{u.get("length") or "—"} aa · '
                f'{html.escape(", ".join(u.get("localization", [])) or "localization unavailable")}</div></div>',
                unsafe_allow_html=True)
        st.markdown("### What the evidence says")
        function_text = (u.get("function") or [g.get("description") or "No concise function annotation was retrieved."])[0]
        st.markdown(f'<div class="gpi-section">{html.escape(function_text)}</div>', unsafe_allow_html=True)
        if pdb:
            st.success(f"🧊 {len(pdb)} experimental structure record(s) are available — open the 3D Structure tab.")
        if a.get("failures"):
            with st.expander("Source warnings"):
                for f in a["failures"]:
                    st.write("• " + f)

    with tabs[1]:
        render_gene(g)
    with tabs[2]:
        render_protein(u)
    with tabs[3]:
        render_clinvar(cv)
        if u.get("disease"):
            st.markdown("### UniProt disease annotations")
            for d in u["disease"]:
                st.write("• " + d)
    with tabs[4]:
        render_pdb(pdb)
    with tabs[5]:
        render_literature(lit)
    with tabs[6]:
        st.markdown("### AI Research Intelligence")
        st.caption("Gemini synthesizes only the evidence retrieved for this search. It is not a substitute for the original records.")
        if st.session_state.get("ai_error"):
            st.warning(st.session_state["ai_error"])
        elif st.session_state.get("report"):
            st.markdown(st.session_state["report"])
        else:
            st.info("No AI synthesis available.")
    with tabs[7]:
        st.markdown("### Evidence sources")
        links = []
        if u.get("url"): links.append(("UniProt " + str(u.get("accession")), u["url"]))
        if g.get("url"): links.append(("NCBI Gene " + str(g.get("gene_id")), g["url"]))
        for x in cv:
            if x.get("url"): links.append(("ClinVar " + str(x.get("accession") or x.get("variation_id")), x["url"]))
        for x in pdb:
            if x.get("url"): links.append(("RCSB PDB " + str(x.get("pdb_id")), x["url"]))
        for x in lit.get("papers", []):
            if x.get("url"): links.append(("PubMed PMID " + str(x.get("pmid")), x["url"]))
        if not links:
            st.info("No source links were returned.")
        else:
            for label, url in links:
                st.link_button(label + " ↗", url)

    st.markdown('<div class="gpi-footer">GPI 3.1 · Live biomedical evidence · Verify important scientific claims against original records.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

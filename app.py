
import os
import re
import html
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests
import streamlit as st
from google import genai
from google.genai import types


APP_TITLE = "GeneProtein Intelligence"
DEFAULT_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT = 20
PUBMED_LIMIT = 8
CLINVAR_LIMIT = 8
PDB_LIMIT = 8

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
NCBI_TOOL = "GeneProteinIntelligence"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return str(value)


def get_config_value(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return value or os.getenv(name, default)


def get_ncbi_api_key() -> str:
    return get_config_value("NCBI_API_KEY", "")


def get_ncbi_email() -> str:
    return get_config_value("NCBI_EMAIL", "")


def http_get(
    url: str,
    params: Dict[str, Any],
    retries: int = 2,
) -> requests.Response:
    headers = {
        "User-Agent": f"{NCBI_TOOL}/2.0",
        "Accept": "application/json, text/plain, */*",
    }

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429 and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after)
                except (TypeError, ValueError):
                    wait_seconds = 2 ** attempt
                time.sleep(min(wait_seconds, 8))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Request failed after retries.")


def http_post_json(
    url: str,
    payload: Dict[str, Any],
    retries: int = 2,
) -> requests.Response:
    headers = {
        "User-Agent": f"{NCBI_TOOL}/2.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429 and attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Request failed after retries.")


def first_ci(mapping: Dict[str, Any], *keys: str) -> Any:
    """Case-insensitive lookup for NCBI/third-party JSON schema changes."""
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def split_aliases(value: Any) -> List[str]:
    if isinstance(value, list):
        raw = [clean_text(x) for x in value]
    else:
        raw = re.split(r"[,;|]", clean_text(value))
    return list(dict.fromkeys(x for x in raw if x))


def extract_uniprot_comments(
    record: Dict[str, Any],
    comment_type: str,
) -> List[str]:
    values = []
    for comment in record.get("comments", []):
        if comment.get("commentType") != comment_type:
            continue
        for text_obj in comment.get("texts", []):
            text = clean_text(text_obj.get("value"))
            if text:
                values.append(text)
    return list(dict.fromkeys(values))


def extract_subcellular_locations(record: Dict[str, Any]) -> List[str]:
    locations = []
    for comment in record.get("comments", []):
        if comment.get("commentType") != "SUBCELLULAR LOCATION":
            continue
        for loc in comment.get("subcellularLocations", []):
            location = loc.get("location", {})
            value = clean_text(location.get("value"))
            if value:
                locations.append(value)
    return list(dict.fromkeys(locations))


def search_uniprot(query: str) -> Dict[str, Any]:
    params = {
        "query": f"({query}) AND (organism_id:9606)",
        "format": "json",
        "size": 1,
    }
    response = http_get(UNIPROT_URL, params)
    results = response.json().get("results", [])
    if not results:
        return {}

    record = results[0]
    genes = record.get("genes", [])
    gene_names = []
    primary_gene = ""

    for gene in genes:
        name = clean_text(gene.get("geneName", {}).get("value"))
        if name and not primary_gene:
            primary_gene = name
        if name:
            gene_names.append(name)

        for alias in gene.get("synonyms", []):
            alias_name = clean_text(alias.get("value"))
            if alias_name:
                gene_names.append(alias_name)

    accession = clean_text(record.get("primaryAccession"))
    organism = clean_text(record.get("organism", {}).get("scientificName"))

    protein_name = clean_text(
        record.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    if not protein_name:
        submitted = record.get("proteinDescription", {}).get("submittedName", [])
        if submitted:
            protein_name = clean_text(
                submitted[0].get("fullName", {}).get("value")
            )

    gene_ids = []
    for ref in record.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GeneID":
            gene_ids.append(clean_text(ref.get("id")))

    sequence = record.get("sequence", {})
    return {
        "accession": accession,
        "protein_name": protein_name,
        "gene_name": primary_gene,
        "gene_aliases": list(dict.fromkeys(gene_names)),
        "organism": organism,
        "length": sequence.get("length"),
        "mass": sequence.get("molWeight"),
        "function": extract_uniprot_comments(record, "FUNCTION"),
        "localization": extract_subcellular_locations(record),
        "disease": extract_uniprot_comments(record, "DISEASE"),
        "ptm": extract_uniprot_comments(record, "PTM"),
        "gene_ids": list(dict.fromkeys(gene_ids)),
        "url": f"https://www.uniprot.org/uniprotkb/{accession}" if accession else "",
    }


def ncbi_common_params() -> Dict[str, Any]:
    params = {"tool": NCBI_TOOL}
    email = get_ncbi_email()
    api_key = get_ncbi_api_key()
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def search_ncbi_gene(query: str) -> Dict[str, Any]:
    term = (
        f'("{query}"[Gene Name] OR "{query}"[Gene Symbol] OR '
        f'"{query}"[All Fields]) AND 9606[Taxonomy ID]'
    )
    search_params = {
        "db": "gene",
        "term": term,
        "retmode": "json",
        "retmax": 3,
        **ncbi_common_params(),
    }
    search_response = http_get(EUTILS_BASE + "esearch.fcgi", search_params)
    ids = search_response.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {}

    summary_params = {
        "db": "gene",
        "id": ",".join(ids),
        "retmode": "json",
        **ncbi_common_params(),
    }
    summary_response = http_get(EUTILS_BASE + "esummary.fcgi", summary_params)
    result = summary_response.json().get("result", {})
    doc = result.get(ids[0], {})
    if not isinstance(doc, dict):
        return {}

    # NCBI ESummary JSON has changed casing/schema across versions.
    symbol = clean_text(first_ci(doc, "Name", "NomenclatureSymbol", "Symbol"))
    description = clean_text(
        first_ci(doc, "Summary", "Description", "DescriptionLong")
    )
    chromosome = clean_text(first_ci(doc, "Chromosome", "chromosome"))
    map_location = clean_text(
        first_ci(doc, "MapLocation", "maplocation", "Maplocation", "Map_Location")
    )

    aliases = []
    for key in (
        "OtherAliases",
        "otheraliases",
        "NomenclatureSymbol",
        "nomenclaturesymbol",
        "Synonym",
        "Synonyms",
    ):
        value = first_ci(doc, key)
        if value:
            aliases.extend(split_aliases(value))
    aliases = [a for a in aliases if a != symbol]

    gene_id = clean_text(first_ci(doc, "uid", "Uid", "GeneID", "GeneId") or ids[0])

    genomic_info = first_ci(doc, "GenomicInfo", "genomicinfo", "GenomicInfoType")
    genomic_summary = ""
    if isinstance(genomic_info, list):
        genomic_summary = f"{len(genomic_info)} genomic placement record(s) returned."
    elif genomic_info:
        genomic_summary = clean_text(genomic_info)

    return {
        "gene_id": gene_id,
        "symbol": symbol,
        "aliases": list(dict.fromkeys(aliases)),
        "description": description,
        "chromosome": chromosome,
        "map_location": map_location,
        "genomic_info": genomic_summary,
        "raw_keys": list(doc.keys()),
        "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
    }


def search_clinvar(query: str, limit: int = CLINVAR_LIMIT) -> List[Dict[str, str]]:
    params = {
        "db": "clinvar",
        "term": f"{query}[gene] AND single_gene[prop]",
        "retmode": "json",
        "retmax": limit,
        "sort": "relevance",
        **ncbi_common_params(),
    }
    response = http_get(EUTILS_BASE + "esearch.fcgi", params)
    ids = response.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    summary_params = {
        "db": "clinvar",
        "id": ",".join(ids),
        "retmode": "json",
        **ncbi_common_params(),
    }
    response = http_get(EUTILS_BASE + "esummary.fcgi", summary_params)
    result = response.json().get("result", {})

    records = []
    for uid in ids:
        doc = result.get(uid, {})
        if not isinstance(doc, dict):
            continue
        accession = clean_text(
            first_ci(doc, "accessionversion", "accession", "rcv_accession", "rcvaccession")
        )
        title = clean_text(
            first_ci(doc, "title", "name", "variation_name", "variationname")
        )
        significance = clean_text(
            first_ci(
                doc,
                "clinical_significance",
                "clinicalsignificance",
                "clinical_significance_description",
                "clinicalsignificancedescription",
            )
        )
        variation_id = clean_text(
            first_ci(doc, "variationid", "variation_id", "uid") or uid
        )
        records.append(
            {
                "uid": uid,
                "accession": accession,
                "title": title or f"ClinVar record {uid}",
                "significance": significance,
                "variation_id": variation_id,
                "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/",
            }
        )
    return records


def search_pubmed(query: str, limit: int = PUBMED_LIMIT) -> List[Dict[str, str]]:
    term = f'("{query}"[Title/Abstract]) AND humans[MeSH Terms]'
    search_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": limit,
        "sort": "relevance",
        **ncbi_common_params(),
    }
    search_response = http_get(EUTILS_BASE + "esearch.fcgi", search_params)
    ids = search_response.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "rettype": "abstract",
        **ncbi_common_params(),
    }
    fetch_response = http_get(EUTILS_BASE + "efetch.fcgi", fetch_params)
    return parse_pubmed_xml(fetch_response.text)


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid = clean_text(article.findtext(".//PMID"))
        title_node = article.find(".//ArticleTitle")
        title = clean_text("".join(title_node.itertext())) if title_node is not None else ""

        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = clean_text("".join(node.itertext()))
            label = clean_text(node.attrib.get("Label"))
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)

        journal = clean_text(article.findtext(".//Journal/Title"))
        year = clean_text(article.findtext(".//PubDate/Year"))
        if not year:
            year = clean_text(article.findtext(".//PubDate/MedlineDate"))[:4]

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = clean_text(author.findtext("LastName"))
            initials = clean_text(author.findtext("Initials"))
            if last:
                authors.append(f"{last} {initials}".strip())

        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": " ".join(abstract_parts),
                "journal": journal,
                "year": year,
                "authors": ", ".join(authors[:6]),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            }
        )
    return papers


def search_pdb(query: str, uniprot_accession: str = "", limit: int = PDB_LIMIT) -> List[Dict[str, Any]]:
    queries = []
    if uniprot_accession:
        queries.append(uniprot_accession)
    queries.append(query)

    ids = []
    for text in queries:
        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": text},
            },
            "return_type": "entry",
            "request_options": {
                "pager": {"start": 0, "rows": limit},
                "results_content_type": ["experimental", "computational"],
            },
        }
        try:
            response = http_post_json(RCSB_SEARCH_URL, payload)
            for item in response.json().get("result_set", []):
                identifier = clean_text(item.get("identifier"))
                if identifier and identifier not in ids:
                    ids.append(identifier)
                if len(ids) >= limit:
                    break
        except requests.RequestException:
            continue
        if len(ids) >= limit:
            break

    structures = []
    for pdb_id in ids[:limit]:
        try:
            response = http_get(f"{RCSB_DATA_URL}/{pdb_id}", {})
            data = response.json()
        except (requests.RequestException, ValueError):
            continue

        entry = data.get("entry", {})
        info = data.get("rcsb_entry_info", {})
        struct = data.get("struct", {})
        title = clean_text(struct.get("title"))

        methods = info.get("experimental_method", [])
        if isinstance(methods, str):
            methods = [methods]
        resolution = info.get("resolution_combined") or []
        if isinstance(resolution, (int, float)):
            resolution = [resolution]

        structures.append(
            {
                "pdb_id": pdb_id,
                "title": title or f"PDB structure {pdb_id}",
                "methods": [clean_text(x) for x in methods if clean_text(x)],
                "resolution": resolution[0] if resolution else None,
                "assembly_count": info.get("assembly_count"),
                "deposition_date": clean_text(entry.get("rcsb_accession_info", {}).get("deposit_date")),
                "url": f"https://www.rcsb.org/structure/{pdb_id}",
            }
        )
    return structures


def build_evidence(
    query: str,
    uniprot: Dict[str, Any],
    ncbi_gene: Dict[str, Any],
    clinvar: List[Dict[str, str]],
    pdb: List[Dict[str, Any]],
    papers: List[Dict[str, str]],
) -> str:
    lines = [
        f"SEARCH TERM: {query}",
        "",
        "SOURCE — UniProt",
        f"URL: {uniprot.get('url', '')}",
        f"Accession: {uniprot.get('accession', '')}",
        f"Protein: {uniprot.get('protein_name', '')}",
        f"Gene: {uniprot.get('gene_name', '')}",
        f"Aliases: {', '.join(uniprot.get('gene_aliases', []))}",
        f"Organism: {uniprot.get('organism', '')}",
        f"Length: {uniprot.get('length', '')} amino acids",
        f"Molecular mass: {uniprot.get('mass', '')} Da",
        f"Function: {' | '.join(uniprot.get('function', []))}",
        f"Localization: {' | '.join(uniprot.get('localization', []))}",
        f"Disease annotations: {' | '.join(uniprot.get('disease', []))}",
        "",
        "SOURCE — NCBI Gene",
        f"URL: {ncbi_gene.get('url', '')}",
        f"Gene ID: {ncbi_gene.get('gene_id', '')}",
        f"Symbol: {ncbi_gene.get('symbol', '')}",
        f"Aliases: {', '.join(ncbi_gene.get('aliases', []))}",
        f"Description: {ncbi_gene.get('description', '')}",
        f"Chromosome: {ncbi_gene.get('chromosome', '')}",
        f"Map location: {ncbi_gene.get('map_location', '')}",
        f"Genomic information: {ncbi_gene.get('genomic_info', '')}",
        "",
        "SOURCE — ClinVar",
    ]

    for item in clinvar:
        lines.extend(
            [
                f"ClinVar accession: {item.get('accession', '')}",
                f"Title: {item.get('title', '')}",
                f"Clinical significance: {item.get('significance', '')}",
                f"Variation ID: {item.get('variation_id', '')}",
                f"URL: {item.get('url', '')}",
                "",
            ]
        )

    lines.append("SOURCE — RCSB Protein Data Bank")
    for item in pdb:
        lines.extend(
            [
                f"PDB ID: {item.get('pdb_id', '')}",
                f"Title: {item.get('title', '')}",
                f"Experimental method: {', '.join(item.get('methods', []))}",
                f"Resolution: {item.get('resolution', '')}",
                f"URL: {item.get('url', '')}",
                "",
            ]
        )

    lines.append("SOURCE — PubMed")
    for i, paper in enumerate(papers, start=1):
        lines.extend(
            [
                f"Paper {i} PMID: {paper.get('pmid', '')}",
                f"Title: {paper.get('title', '')}",
                f"Journal: {paper.get('journal', '')}",
                f"Year: {paper.get('year', '')}",
                f"Authors: {paper.get('authors', '')}",
                f"URL: {paper.get('url', '')}",
                f"Abstract: {paper.get('abstract', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def generate_ai_report(evidence: str, model_name: str) -> str:
    api_key = get_config_value("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets."
        )

    client = genai.Client(api_key=api_key)
    system_instruction = """
You are GeneProtein Intelligence (GPI), a biomedical research assistant.

Use ONLY the retrieved source material supplied by the user. Never invent facts,
numbers, variants, diseases, mechanisms, citations, or research findings.

If a requested fact is not supported, say:
"Not available in the retrieved sources."

Rules:
- Distinguish association from causation.
- Do not diagnose or recommend treatment.
- Preserve uncertainty and conflicting evidence.
- Do not imply the retrieved PubMed set is exhaustive.
- Do not claim to have read full papers when only abstracts were retrieved.
- Identify PMIDs when discussing literature.
- Clearly label evidence by source: [UniProt], [NCBI Gene], [ClinVar],
  [RCSB PDB], [PubMed PMID: ...].
- For ClinVar, report the database's classification as source evidence; do not
  convert it into patient-specific clinical advice.
- For PDB, distinguish experimental structures from computed structure models.

Return:
1. Executive Overview
2. Gene & Genomic Context
3. Protein Function & Localization
4. Disease & Variant Evidence
5. 3D Structural Evidence
6. Literature Findings
7. Research Intelligence
8. Evidence Limitations
9. Sources
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=evidence,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=3500,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text.")
    return text


def source_link(label: str, url: str, text: str) -> str:
    return f"- **{label}:** [{html.escape(text)}]({url})"


def render_sources(uniprot, ncbi_gene, clinvar, pdb, papers) -> None:
    st.subheader("Evidence Sources")
    if uniprot.get("url"):
        st.markdown(source_link("UniProt", uniprot["url"], uniprot.get("accession", "record")))
    if ncbi_gene.get("url"):
        st.markdown(source_link("NCBI Gene", ncbi_gene["url"], f"Gene {ncbi_gene.get('gene_id', '')}"))
    for item in clinvar:
        if item.get("url"):
            st.markdown(source_link("ClinVar", item["url"], item.get("accession") or item.get("title", "record")))
    for item in pdb:
        if item.get("url"):
            st.markdown(source_link("RCSB PDB", item["url"], item.get("pdb_id", "structure")))
    for paper in papers:
        if paper.get("url"):
            st.markdown(source_link("PubMed", paper["url"], f"PMID {paper.get('pmid', '')}"))


def render_gene(ncbi_gene: Dict[str, Any], uniprot: Dict[str, Any]) -> None:
    st.subheader("Gene & Genomic Context")
    if not ncbi_gene:
        st.info("No NCBI Gene record was found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gene ID", ncbi_gene.get("gene_id") or "—")
    c2.metric("Chromosome", ncbi_gene.get("chromosome") or "—")
    c3.metric("Map location", ncbi_gene.get("map_location") or "—")
    c4.metric("Symbol", ncbi_gene.get("symbol") or uniprot.get("gene_name") or "—")

    st.markdown("**Aliases**")
    st.write(", ".join(ncbi_gene.get("aliases", [])) or "Not available in the retrieved NCBI record.")

    st.markdown("**Gene description**")
    st.write(ncbi_gene.get("description") or "Not available in the retrieved NCBI record.")

    st.markdown("**Genomic information**")
    st.write(ncbi_gene.get("genomic_info") or "No additional genomic placement summary was returned.")


def render_protein(uniprot: Dict[str, Any]) -> None:
    st.subheader("Protein Intelligence")
    if not uniprot:
        st.info("No human UniProt record was found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("UniProt", uniprot.get("accession") or "—")
    c2.metric("Length", f"{uniprot.get('length')} aa" if uniprot.get("length") else "—")
    c3.metric("Mass", f"{uniprot.get('mass'):,} Da" if isinstance(uniprot.get("mass"), int) else "—")
    c4.metric("Gene", uniprot.get("gene_name") or "—")

    st.markdown("**Protein name**")
    st.write(uniprot.get("protein_name") or "Not available.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Function**")
        st.write("\n\n".join(uniprot.get("function", [])) or "Not available in the retrieved UniProt record.")
    with c2:
        st.markdown("**Subcellular localization**")
        st.write(", ".join(uniprot.get("localization", [])) or "Not available in the retrieved UniProt record.")

    if uniprot.get("gene_aliases"):
        st.markdown("**Gene names / aliases from UniProt**")
        st.write(", ".join(uniprot["gene_aliases"]))


def render_diseases(clinvar, uniprot) -> None:
    st.subheader("Disease & Variant Evidence")

    if uniprot.get("disease"):
        st.markdown("### UniProt disease annotations")
        for item in uniprot["disease"]:
            st.write(f"- {item}")
    else:
        st.info("No UniProt disease annotation was retrieved for this record.")

    st.markdown("### ClinVar evidence")
    if not clinvar:
        st.write("No ClinVar records were retrieved for this gene search.")
        return

    st.caption(
        f"{len(clinvar)} ClinVar records retrieved. These are database records and "
        "should not be interpreted as patient-specific medical advice."
    )
    for item in clinvar:
        title = item.get("title") or "ClinVar record"
        with st.expander(title):
            st.write(f"**Accession:** {item.get('accession') or '—'}")
            st.write(f"**Clinical significance:** {item.get('significance') or 'Not reported in the retrieved summary.'}")
            st.write(f"**Variation ID:** {item.get('variation_id') or '—'}")
            if item.get("url"):
                st.markdown(f"[Open ClinVar record]({item['url']})")


def render_pdb(pdb) -> None:
    st.subheader("3D Structural Evidence")
    if not pdb:
        st.info("No RCSB Protein Data Bank structures were retrieved for this search.")
        st.caption("A missing PDB result does not mean that the protein has no known structure; it means no matching structures were returned by the current search.")
        return

    st.metric("Matching structures", len(pdb))
    for item in pdb:
        title = item.get("title") or item.get("pdb_id")
        with st.expander(f"{item.get('pdb_id')} — {title}"):
            st.write(f"**Experimental method:** {', '.join(item.get('methods', [])) or 'Not reported'}")
            resolution = item.get("resolution")
            st.write(f"**Resolution:** {resolution} Å" if resolution else "**Resolution:** Not reported")
            if item.get("deposition_date"):
                st.write(f"**Deposit date:** {item['deposition_date']}")
            st.markdown(f"[View structure on RCSB PDB]({item['url']})")


def render_literature(papers) -> None:
    st.subheader("Scientific Literature")
    if not papers:
        st.info("No relevant PubMed results were returned.")
        return
    st.caption(f"Showing {len(papers)} relevant PubMed records; this is not an exhaustive literature review.")
    for paper in papers:
        with st.expander(paper.get("title") or "Untitled article"):
            st.write(
                f"**Journal:** {paper.get('journal') or '—'}  \n"
                f"**Year:** {paper.get('year') or '—'}  \n"
                f"**PMID:** {paper.get('pmid') or '—'}"
            )
            if paper.get("authors"):
                st.write(f"**Authors:** {paper['authors']}")
            st.write(paper.get("abstract") or "Abstract unavailable.")
            if paper.get("url"):
                st.markdown(f"[Open PubMed record]({paper['url']})")


def run_analysis(query: str) -> None:
    sources = {}
    failures = []

    with st.status("Building your evidence profile…", expanded=True) as status:
        steps = [
            ("UniProt", lambda: search_uniprot(query)),
            ("NCBI Gene", lambda: search_ncbi_gene(query)),
        ]

        for label, func in steps:
            st.write(f"Retrieving {label}…")
            try:
                sources["uniprot" if label == "UniProt" else "ncbi_gene"] = func()
            except Exception as exc:
                sources["uniprot" if label == "UniProt" else "ncbi_gene"] = {}
                failures.append(f"{label}: {exc}")

        gene_symbol = sources.get("ncbi_gene", {}).get("symbol") or query
        st.write("Retrieving ClinVar…")
        try:
            sources["clinvar"] = search_clinvar(gene_symbol)
        except Exception as exc:
            sources["clinvar"] = []
            failures.append(f"ClinVar: {exc}")

        st.write("Searching RCSB Protein Data Bank…")
        try:
            sources["pdb"] = search_pdb(
                gene_symbol,
                sources.get("uniprot", {}).get("accession", ""),
            )
        except Exception as exc:
            sources["pdb"] = []
            failures.append(f"RCSB PDB: {exc}")

        st.write("Retrieving PubMed literature…")
        try:
            sources["papers"] = search_pubmed(gene_symbol)
        except Exception as exc:
            sources["papers"] = []
            failures.append(f"PubMed: {exc}")

        status.update(label="Evidence retrieval complete", state="complete")

    if failures:
        with st.expander("Some sources were unavailable", expanded=False):
            for failure in failures:
                st.write(f"- {failure}")

    if not any(sources.get(k) for k in ("uniprot", "ncbi_gene", "clinvar", "pdb", "papers")):
        st.error("No usable evidence was retrieved. Try a human gene such as TP53, BRCA1, or EGFR.")
        return

    st.session_state["analysis"] = {
        "query": query,
        **sources,
        "failures": failures,
    }

    model_name = get_config_value("GEMINI_MODEL", DEFAULT_MODEL)
    evidence = build_evidence(
        query,
        sources.get("uniprot", {}),
        sources.get("ncbi_gene", {}),
        sources.get("clinvar", []),
        sources.get("pdb", []),
        sources.get("papers", []),
    )

    with st.spinner(f"Gemini is synthesizing the evidence using {model_name}…"):
        try:
            st.session_state["report"] = generate_ai_report(evidence, model_name)
            st.session_state["ai_error"] = ""
        except Exception as exc:
            st.session_state["report"] = ""
            st.session_state["ai_error"] = str(exc)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px;}
        .gpi-hero {
            padding: 2.1rem 2.2rem;
            border: 1px solid rgba(120,140,180,.25);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(35,55,90,.12), rgba(70,120,150,.06));
            margin-bottom: 1.2rem;
        }
        .gpi-kicker {font-size:.82rem; letter-spacing:.12em; text-transform:uppercase; opacity:.7; font-weight:700;}
        .gpi-title {font-size:2.6rem; line-height:1.05; font-weight:800; margin:.25rem 0 .6rem;}
        .gpi-subtitle {font-size:1.05rem; opacity:.78; max-width:850px;}
        .source-chip {
            display:inline-block; padding:.3rem .65rem; border-radius:999px;
            border:1px solid rgba(120,140,180,.28); margin:.15rem .25rem .15rem 0;
            font-size:.78rem;
        }
        .section-card {
            padding:1.05rem 1.15rem; border:1px solid rgba(120,140,180,.2);
            border-radius:18px; background:rgba(128,128,128,.045); height:100%;
        }
        div[data-testid="stMetric"] {
            border:1px solid rgba(120,140,180,.2); padding:.75rem; border-radius:16px;
            background:rgba(128,128,128,.035);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    st.markdown(
        """
        <div class="gpi-hero">
          <div class="gpi-kicker">Biomedical AI Research Workspace</div>
          <div class="gpi-title">🧬 GeneProtein Intelligence</div>
          <div class="gpi-subtitle">
            Evidence-grounded exploration of human genes, proteins, disease evidence,
            3D structures and scientific literature.
          </div>
          <div style="margin-top:.8rem;">
            <span class="source-chip">UniProt</span>
            <span class="source-chip">NCBI Gene</span>
            <span class="source-chip">ClinVar</span>
            <span class="source-chip">RCSB PDB</span>
            <span class="source-chip">PubMed</span>
            <span class="source-chip">Gemini AI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### GPI")
        st.write("Research-oriented gene & protein intelligence.")
        st.divider()
        st.markdown("**Try:** `TP53` · `BRCA1` · `EGFR`")
        st.caption("Educational/research tool. Not a diagnostic or treatment system.")
        if st.button("Clear current analysis", use_container_width=True):
            for key in ("analysis", "report", "ai_error"):
                st.session_state.pop(key, None)
            st.rerun()

    query = st.text_input(
        "Search a human gene or protein",
        placeholder="e.g. BRCA1, TP53, EGFR",
        max_chars=100,
        label_visibility="collapsed",
    )

    if st.button("🔎  Analyze gene / protein", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Enter a gene or protein name first.")
        elif len(query.strip()) < 2:
            st.error("Please enter at least 2 characters.")
        else:
            run_analysis(query.strip())

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.markdown(
            """
            <div class="section-card">
            <b>Start a research profile</b><br>
            Search a human gene or protein to retrieve structured evidence from
            multiple biomedical databases and generate an AI-grounded synthesis.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    uniprot = analysis.get("uniprot", {})
    ncbi_gene = analysis.get("ncbi_gene", {})
    clinvar = analysis.get("clinvar", [])
    pdb = analysis.get("pdb", [])
    papers = analysis.get("papers", [])

    symbol = ncbi_gene.get("symbol") or uniprot.get("gene_name") or analysis["query"]
    protein = uniprot.get("protein_name") or "Protein record unavailable"

    st.markdown(f"## {symbol}")
    st.caption(protein)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gene ID", ncbi_gene.get("gene_id") or "—")
    c2.metric("UniProt", uniprot.get("accession") or "—")
    c3.metric("ClinVar", len(clinvar))
    c4.metric("PDB structures", len(pdb))
    c5.metric("PubMed", len(papers))

    tabs = st.tabs(
        ["Overview", "Gene", "Protein", "Diseases & Variants", "3D Structure", "Literature", "AI Research", "Sources"]
    )

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card"><b>Gene context</b><br><br>'
                        f"Chromosome: <b>{ncbi_gene.get('chromosome') or '—'}</b><br>"
                        f"Map location: <b>{ncbi_gene.get('map_location') or '—'}</b><br>"
                        f"Aliases: {', '.join(ncbi_gene.get('aliases', [])) or '—'}"
                        "</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-card"><b>Protein snapshot</b><br><br>'
                        f"Name: <b>{html.escape(protein)}</b><br>"
                        f"Length: <b>{uniprot.get('length') or '—'} aa</b><br>"
                        f"Localization: {', '.join(uniprot.get('localization', [])) or '—'}"
                        "</div>", unsafe_allow_html=True)

        st.markdown("### What the retrieved evidence says")
        st.write(
            uniprot.get("function", ["No UniProt function annotation was retrieved."])[0]
        )

    with tabs[1]:
        render_gene(ncbi_gene, uniprot)

    with tabs[2]:
        render_protein(uniprot)

    with tabs[3]:
        render_diseases(clinvar, uniprot)

    with tabs[4]:
        render_pdb(pdb)

    with tabs[5]:
        render_literature(papers)

    with tabs[6]:
        st.subheader("AI Research Intelligence")
        ai_error = st.session_state.get("ai_error", "")
        report = st.session_state.get("report", "")
        if ai_error:
            st.warning(f"AI synthesis is unavailable right now: {ai_error}")
            st.info("The retrieved database evidence is still available in the other tabs.")
        elif report:
            st.markdown(report)
        else:
            st.info("No AI synthesis is available.")

    with tabs[7]:
        render_sources(uniprot, ncbi_gene, clinvar, pdb, papers)

    st.divider()
    st.caption(
        "GPI v2 is an educational/research prototype. Database records and AI synthesis "
        "should be verified against the original sources for important scientific work."
    )


if __name__ == "__main__":
    main()

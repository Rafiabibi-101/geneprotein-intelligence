import os
import re
import html
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

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NCBI_TOOL = "GeneProteinIntelligence"
# NCBI credentials are read through get_config_value() below.
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")


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


def http_get(
    url: str,
    params: Dict[str, Any],
    retries: int = 2,
) -> requests.Response:
    headers = {"User-Agent": f"{NCBI_TOOL}/1.0"}

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


def extract_uniprot_comments(record: Dict[str, Any], comment_type: str) -> List[str]:
    values = []
    for comment in record.get("comments", []):
        if comment.get("commentType") != comment_type:
            continue
        for text_obj in comment.get("texts", []):
            text = clean_text(text_obj.get("value"))
            if text:
                values.append(text)
    return values


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
        "fields": (
            "accession,id,protein_name,gene_names,organism_name,length,"
            "cc_function,cc_subcellular_location,cc_disease,cc_ptm,xref_geneid"
        ),
    }

    response = http_get(UNIPROT_URL, params)
    data = response.json()
    results = data.get("results", [])
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

    return {
        "accession": accession,
        "protein_name": protein_name,
        "gene_name": primary_gene,
        "gene_aliases": list(dict.fromkeys(gene_names)),
        "organism": organism,
        "length": record.get("sequence", {}).get("length"),
        "function": extract_uniprot_comments(record, "FUNCTION"),
        "localization": extract_subcellular_locations(record),
        "disease": extract_uniprot_comments(record, "DISEASE"),
        "ptm": extract_uniprot_comments(record, "PTM"),
        "gene_ids": list(dict.fromkeys(gene_ids)),
        "url": (
            f"https://www.uniprot.org/uniprotkb/{accession}"
            if accession
            else ""
        ),
    }


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
        "tool": NCBI_TOOL,
    }
    if NCBI_EMAIL:
        search_params["email"] = NCBI_EMAIL
    if get_ncbi_api_key():
        search_params["api_key"] = get_ncbi_api_key()

    search_response = http_get(
        EUTILS_BASE + "esearch.fcgi",
        search_params,
    )
    ids = search_response.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return {}

    summary_params = {
        "db": "gene",
        "id": ",".join(ids),
        "retmode": "json",
        "tool": NCBI_TOOL,
    }
    if NCBI_EMAIL:
        summary_params["email"] = NCBI_EMAIL
    if get_ncbi_api_key():
        summary_params["api_key"] = get_ncbi_api_key()

    summary_response = http_get(
        EUTILS_BASE + "esummary.fcgi",
        summary_params,
    )
    result = summary_response.json().get("result", {})
    doc = result.get(ids[0], {})

    aliases = []
    for key in ("OtherAliases", "NomenclatureSymbol"):
        value = clean_text(doc.get(key))
        if value:
            aliases.extend(
                [x.strip() for x in re.split(r"[,;]", value) if x.strip()]
            )

    gene_id = clean_text(doc.get("uid") or ids[0])

    return {
        "gene_id": gene_id,
        "symbol": clean_text(doc.get("Name")),
        "aliases": list(dict.fromkeys(aliases)),
        "description": clean_text(doc.get("Summary")),
        "chromosome": clean_text(doc.get("Chromosome")),
        "map_location": clean_text(doc.get("MapLocation")),
        "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
    }


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_text)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        pmid = clean_text(article.findtext(".//PMID"))

        title_node = article.find(".//ArticleTitle")
        title = (
            clean_text("".join(title_node.itertext()))
            if title_node is not None
            else ""
        )

        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = clean_text("".join(node.itertext()))
            label = clean_text(node.attrib.get("Label"))
            abstract_parts.append(
                f"{label}: {text}" if label else text
            )

        journal = clean_text(article.findtext(".//Journal/Title"))
        year = clean_text(article.findtext(".//PubDate/Year"))
        if not year:
            year = clean_text(
                article.findtext(".//PubDate/MedlineDate")
            )[:4]

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
                "abstract": " ".join(
                    x for x in abstract_parts if x
                ),
                "journal": journal,
                "year": year,
                "authors": ", ".join(authors[:6]),
                "url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    if pmid
                    else ""
                ),
            }
        )

    return papers


def search_pubmed(
    query: str,
    limit: int = PUBMED_LIMIT,
) -> List[Dict[str, str]]:
    term = f'("{query}"[Title/Abstract]) AND humans[MeSH Terms]'

    search_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": limit,
        "sort": "relevance",
        "tool": NCBI_TOOL,
    }
    if NCBI_EMAIL:
        search_params["email"] = NCBI_EMAIL
    if get_ncbi_api_key():
        search_params["api_key"] = get_ncbi_api_key()

    search_response = http_get(
        EUTILS_BASE + "esearch.fcgi",
        search_params,
    )
    ids = search_response.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "rettype": "abstract",
        "tool": NCBI_TOOL,
    }
    if NCBI_EMAIL:
        fetch_params["email"] = NCBI_EMAIL
    if get_ncbi_api_key():
        fetch_params["api_key"] = get_ncbi_api_key()

    fetch_response = http_get(
        EUTILS_BASE + "efetch.fcgi",
        fetch_params,
    )

    return parse_pubmed_xml(fetch_response.text)


def build_evidence(
    query: str,
    uniprot: Dict[str, Any],
    ncbi_gene: Dict[str, Any],
    papers: List[Dict[str, str]],
) -> str:
    lines = [
        f"SEARCH TERM: {query}",
        "",
        "SOURCE 1 — UniProt",
        f"URL: {uniprot.get('url', '')}",
        f"Accession: {uniprot.get('accession', '')}",
        f"Protein: {uniprot.get('protein_name', '')}",
        f"Gene: {uniprot.get('gene_name', '')}",
        f"Aliases: {', '.join(uniprot.get('gene_aliases', []))}",
        f"Organism: {uniprot.get('organism', '')}",
        f"Length: {uniprot.get('length', '')} amino acids",
        f"Function: {' | '.join(uniprot.get('function', []))}",
        f"Localization: {' | '.join(uniprot.get('localization', []))}",
        f"Disease annotations: {' | '.join(uniprot.get('disease', []))}",
        f"PTM annotations: {' | '.join(uniprot.get('ptm', []))}",
        f"Gene IDs: {', '.join(uniprot.get('gene_ids', []))}",
        "",
        "SOURCE 2 — NCBI Gene",
        f"URL: {ncbi_gene.get('url', '')}",
        f"Gene ID: {ncbi_gene.get('gene_id', '')}",
        f"Symbol: {ncbi_gene.get('symbol', '')}",
        f"Aliases: {', '.join(ncbi_gene.get('aliases', []))}",
        f"Description: {ncbi_gene.get('description', '')}",
        f"Chromosome: {ncbi_gene.get('chromosome', '')}",
        f"Map location: {ncbi_gene.get('map_location', '')}",
        "",
        "SOURCE 3 — PubMed literature",
    ]

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


def generate_ai_report(
    evidence: str,
    model_name: str,
) -> str:
    api_key = get_config_value("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets "
            "or your local environment."
        )

    client = genai.Client(api_key=api_key)

    system_instruction = """
You are GeneProtein Intelligence (GPI), a biomedical research assistant.

Use ONLY the retrieved source material supplied in the user message. Do not use
unstated background knowledge to fill gaps. Never invent facts, numbers,
variants, diseases, mechanisms, citations, or research findings.

If a requested fact is not supported by the supplied evidence, write:
"Not available in the retrieved sources."

Scientific rules:
- Distinguish association from causation.
- Do not make clinical diagnoses, treatment recommendations, or patient-specific advice.
- Preserve uncertainty and conflicting evidence.
- Do not imply that the retrieved PubMed set represents all literature.
- Do not claim to have read papers whose abstracts/text were not supplied.
- When discussing literature, identify the PMID when available.
- Keep terminology scientifically accurate and readable for life-science students.
- Use source labels such as [UniProt], [NCBI Gene], and [PubMed PMID: ...].

Return these sections:
1. Gene / Protein Overview
2. Protein Function
3. Subcellular Localization
4. Gene Information
5. Disease Associations
6. Mutations / Variants
7. Literature Findings
8. Research Intelligence
9. Evidence Limitations
10. Source References

In "Research Intelligence", only report patterns or insights that can reasonably
be supported by the retrieved evidence. Do not speculate beyond it.
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=evidence,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=3000,
        ),
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text.")

    return text


def render_sources(
    uniprot: Dict[str, Any],
    ncbi_gene: Dict[str, Any],
    papers: List[Dict[str, str]],
) -> None:
    st.subheader("Sources")

    if uniprot.get("url"):
        st.markdown(
            f"- **UniProt:** "
            f"[{uniprot.get('accession', 'record')}]({uniprot['url']})"
        )

    if ncbi_gene.get("url"):
        st.markdown(
            f"- **NCBI Gene:** "
            f"[Gene {ncbi_gene.get('gene_id', '')}]({ncbi_gene['url']})"
        )

    for paper in papers:
        if paper.get("url"):
            title = html.escape(
                paper.get("title", "PubMed article")
            )
            st.markdown(
                f"- **PubMed:** [{title}]({paper['url']})"
            )


def render_uniprot(uniprot: Dict[str, Any]) -> None:
    st.subheader("Protein Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "UniProt accession",
            uniprot.get("accession") or "Unavailable",
        )
        st.write(
            f"**Protein:** "
            f"{uniprot.get('protein_name') or 'Unavailable'}"
        )
        st.write(
            f"**Gene:** "
            f"{uniprot.get('gene_name') or 'Unavailable'}"
        )
        st.write(
            f"**Organism:** "
            f"{uniprot.get('organism') or 'Unavailable'}"
        )

    with col2:
        length = uniprot.get("length")
        st.metric(
            "Protein length",
            f"{length} aa" if length else "Unavailable",
        )
        st.write(
            f"**Gene aliases:** "
            f"{', '.join(uniprot.get('gene_aliases', [])) or 'Unavailable'}"
        )

    st.markdown("**Function**")
    st.write(
        "\n\n".join(uniprot.get("function", []))
        or "Unavailable in the retrieved UniProt record."
    )

    st.markdown("**Subcellular localization**")
    st.write(
        ", ".join(uniprot.get("localization", []))
        or "Unavailable in the retrieved UniProt record."
    )

    if uniprot.get("disease"):
        st.markdown("**UniProt disease annotations**")
        for item in uniprot["disease"]:
            st.write(f"- {item}")


def render_gene(
    ncbi_gene: Dict[str, Any],
    uniprot: Dict[str, Any],
) -> None:
    st.subheader("Gene Information")

    if not ncbi_gene:
        st.info("No NCBI Gene record was found for this search.")
        return

    st.write(
        f"**Gene ID:** "
        f"{ncbi_gene.get('gene_id') or 'Unavailable'}"
    )
    st.write(
        f"**Symbol:** "
        f"{ncbi_gene.get('symbol') or uniprot.get('gene_name') or 'Unavailable'}"
    )
    st.write(
        f"**Aliases:** "
        f"{', '.join(ncbi_gene.get('aliases', [])) or 'Unavailable'}"
    )
    st.write(
        f"**Chromosome:** "
        f"{ncbi_gene.get('chromosome') or 'Unavailable'}"
    )
    st.write(
        f"**Map location:** "
        f"{ncbi_gene.get('map_location') or 'Unavailable'}"
    )

    st.markdown("**NCBI description**")
    st.write(
        ncbi_gene.get("description")
        or "Unavailable in the retrieved NCBI Gene record."
    )


def render_literature(
    papers: List[Dict[str, str]],
) -> None:
    st.subheader("Scientific Literature")

    if not papers:
        st.info(
            "No relevant PubMed results were returned for this search."
        )
        return

    st.caption(
        f"Showing {len(papers)} relevant PubMed records retrieved for this "
        "search. This is not an exhaustive literature review."
    )

    for paper in papers:
        title = paper.get("title") or "Untitled article"

        with st.expander(title):
            st.write(
                f"**Journal:** {paper.get('journal') or 'Unavailable'}  \n"
                f"**Year:** {paper.get('year') or 'Unavailable'}  \n"
                f"**PMID:** {paper.get('pmid') or 'Unavailable'}"
            )

            if paper.get("authors"):
                st.write(f"**Authors:** {paper['authors']}")

            st.write(
                paper.get("abstract")
                or "Abstract unavailable."
            )

            if paper.get("url"):
                st.markdown(
                    f"[Open PubMed record]({paper['url']})"
                )


def run_analysis(query: str) -> None:
    with st.spinner(
        "Retrieving UniProt, NCBI Gene and PubMed evidence..."
    ):
        try:
            uniprot = search_uniprot(query)
            ncbi_gene = search_ncbi_gene(query)
            papers = search_pubmed(query)

        except requests.RequestException as exc:
            st.error(f"A data-source request failed: {exc}")
            return

        except (ValueError, ET.ParseError) as exc:
            st.error(
                f"A data-source response could not be parsed: {exc}"
            )
            return

        except Exception as exc:
            st.error(f"Unexpected data retrieval error: {exc}")
            return

    if not uniprot and not ncbi_gene and not papers:
        st.warning(
            "No human UniProt, NCBI Gene, or PubMed results were found. "
            "Try a gene/protein such as TP53, BRCA1, or EGFR."
        )
        return

    st.session_state["analysis"] = {
        "query": query,
        "uniprot": uniprot,
        "ncbi_gene": ncbi_gene,
        "papers": papers,
    }

    model_name = get_config_value(
        "GEMINI_MODEL",
        DEFAULT_MODEL,
    )

    with st.spinner(
        f"Gemini is synthesizing the retrieved evidence using {model_name}..."
    ):
        try:
            evidence = build_evidence(
                query,
                uniprot,
                ncbi_gene,
                papers,
            )
            report = generate_ai_report(
                evidence,
                model_name,
            )
            st.session_state["report"] = report

        except Exception as exc:
            st.session_state["report"] = ""
            st.error(f"AI synthesis failed: {exc}")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧬 GeneProtein Intelligence")
    st.caption(
        "AI-powered biomedical research assistant for gene and protein exploration."
    )

    with st.sidebar:
        st.header("About GPI")
        st.write(
            "GPI retrieves human gene/protein evidence from UniProt, NCBI Gene, "
            "and PubMed, then uses Gemini Flash to organize the retrieved evidence "
            "into a research-oriented profile."
        )
        st.warning(
            "Educational/research tool only. Not a diagnostic or treatment system."
        )
        st.divider()
        st.write("**Suggested tests:** TP53 · BRCA1 · EGFR")

    st.markdown(
        "Enter a **human gene or protein name**. GPI will retrieve source evidence "
        "and generate a grounded research summary."
    )

    query = st.text_input(
        "Gene / protein",
        placeholder="e.g. TP53, BRCA1, EGFR",
        max_chars=100,
    ).strip()

    if st.button(
        "🔎 Analyze",
        type="primary",
        use_container_width=True,
    ):
        if not query:
            st.error("Please enter a gene or protein name.")
        elif len(query) < 2:
            st.error("Please enter at least 2 characters.")
        else:
            run_analysis(query)

    analysis = st.session_state.get("analysis")

    if not analysis:
        st.info(
            "Enter a gene/protein above and click Analyze."
        )
        return

    uniprot = analysis["uniprot"]
    ncbi_gene = analysis["ncbi_gene"]
    papers = analysis["papers"]

    st.divider()
    st.header(
        f"Research Profile: {analysis['query']}"
    )

    tabs = st.tabs(
        [
            "Overview",
            "Protein",
            "Gene",
            "Diseases",
            "Literature",
            "AI Research Intelligence",
            "Sources",
        ]
    )

    with tabs[0]:
        st.subheader("Gene / Protein Overview")
        st.write(
            f"**Search term:** {analysis['query']}"
        )
        st.write(
            f"**Protein:** "
            f"{uniprot.get('protein_name') or 'Unavailable'}"
        )
        st.write(
            f"**Gene:** "
            f"{uniprot.get('gene_name') or ncbi_gene.get('symbol') or 'Unavailable'}"
        )
        st.write(
            f"**Organism:** "
            f"{uniprot.get('organism') or 'Human result not available'}"
        )
        st.write(
            uniprot.get(
                "function",
                ["No UniProt function annotation retrieved."],
            )[0]
        )

    with tabs[1]:
        render_uniprot(uniprot)

    with tabs[2]:
        render_gene(ncbi_gene, uniprot)

    with tabs[3]:
        st.subheader("Disease Associations")

        disease_items = uniprot.get("disease", [])

        if disease_items:
            st.info(
                "These are annotations retrieved from UniProt. They are presented "
                "as source information, not as individual medical advice or proof "
                "of causation."
            )
            for item in disease_items:
                st.write(f"- {item}")
        else:
            st.write(
                "No UniProt disease annotations were retrieved for this record."
            )

        st.write(
            "Clinical variant interpretation is intentionally limited in the MVP. "
            "Future versions can integrate ClinVar/Open Targets with explicit source handling."
        )

    with tabs[4]:
        render_literature(papers)

    with tabs[5]:
        st.subheader("AI Research Intelligence")
        report = st.session_state.get("report", "")

        if report:
            st.markdown(report)
        else:
            st.info(
                "No AI report is available. Check the Gemini API key "
                "and error message above."
            )

    with tabs[6]:
        render_sources(
            uniprot,
            ncbi_gene,
            papers,
        )

    st.divider()
    st.caption(
        "GPI is an educational/research prototype. AI synthesis is based only "
        "on the retrieved source set shown in this application. Always verify "
        "important findings against the original records and publications."
    )


if __name__ == "__main__":
    main()





       
           
        
    

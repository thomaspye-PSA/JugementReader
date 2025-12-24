import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from lxml import etree

# --- Namespaces ---
NS_AKN = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
NS_UK = {"uk": "https://caselaw.nationalarchives.gov.uk/akn"}

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def iter_case_dirs(repo_root: str) -> List[str]:
    """
    Finds all judgment directories containing meta.json and data.xml.
    """
    case_dirs = []
    for root, dirs, files in os.walk(repo_root):
        if "meta.json" in files and "data.xml" in files:
            case_dirs.append(root)
    return case_dirs

def load_meta(case_dir: str) -> Dict[str, Any]:
    with open(os.path.join(case_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def parse_xml(case_dir: str) -> etree._Element:
    xml_path = os.path.join(case_dir, "data.xml")
    parser = etree.XMLParser(recover=True, huge_tree=True)
    with open(xml_path, "rb") as f:
        return etree.fromstring(f.read(), parser=parser)

def extract_case_identity(root: etree._Element) -> Dict[str, Optional[str]]:
    """
    Pull stable identifiers for the case from FRBR + uk:cite if present.
    """
    # FRBRExpression/FRBRthis gives the public TNA case page
    frbr_this = root.xpath(
        "string(.//*[local-name()='FRBRExpression']/*[local-name()='FRBRthis'][1]/@value)"
    )
    frbr_uri = root.xpath(
        "string(.//*[local-name()='FRBRExpression']/*[local-name()='FRBRuri'][1]/@value)"
    )

    # UK proprietary cite if present
    neutral = root.xpath("string(.//*[local-name()='proprietary']//*[local-name()='cite'][1])")

    # Work name (often case name)
    work_name = root.xpath(
        "string(.//*[local-name()='FRBRWork']/*[local-name()='FRBRname'][1]/@value)"
    )

    return {
        "tna_page": _clean_text(frbr_this) or None,
        "tna_uri": _clean_text(frbr_uri) or None,
        "neutral_citation": _clean_text(neutral) or None,
        "work_name": _clean_text(work_name) or None,
    }

def extract_paragraph_index(root: etree._Element) -> Dict[str, Tuple[Optional[str], Optional[int], str]]:
    """
    Map paragraph eId -> (num_text, num_int, full_text)
    Only uses main body paragraphs with eId like para_123.
    """
    out = {}
    paras = root.xpath(".//*[local-name()='judgmentBody']//*[local-name()='paragraph'][@eId]")
    for p in paras:
        eId = p.get("eId")
        if not eId or not eId.startswith("para_"):
            continue

        num_txt = _clean_text("".join(p.xpath("./*[local-name()='num'][1]//text()")))
        num_int = None
        m = re.search(r"(\d+)", num_txt or "")
        if m:
            num_int = int(m.group(1))

        content_node = p.xpath("./*[local-name()='content'][1]")
        if content_node:
            p_text = _clean_text(" ".join(content_node[0].xpath(".//text()")))
        else:
            p_text = _clean_text(" ".join(p.xpath(".//text()")))

        out[eId] = (num_txt or None, num_int, p_text)

    return out

def extract_case_refs(root: etree._Element) -> List[Dict[str, Any]]:
    """
    Extract <ref> elements that likely refer to other cases.
    Returns list of dicts with href, canonical, text, para_eid.
    """
    refs = root.xpath(".//*[local-name()='judgmentBody']//*[local-name()='ref']")
    out = []
    for r in refs:
        href = (r.get("href") or "").strip()

        # uk:type attribute (namespaced) sometimes indicates case/legislation
        uk_type = None
        for k, v in r.attrib.items():
            if k.endswith("}type"):  # catches {uk-namespace}type
                uk_type = v

        canonical = None
        for k, v in r.attrib.items():
            if k.endswith("}canonical"):
                canonical = v

        txt = _clean_text(" ".join(r.xpath(".//text()")))
        if not txt and canonical:
            txt = canonical

        # Filter: keep only case refs
        # Primary: uk:type == "case"
        # Secondary heuristic: canonical looks like [2024] EWCA Civ 419 etc
        looks_like_case = False
        if canonical and re.search(r"\[\d{4}\]\s+[A-Z]{2,}.*\d+", canonical):
            looks_like_case = True

        if uk_type == "case" or looks_like_case:
            # find containing paragraph eId (walk up ancestors)
            para_eId = None
            for anc in r.iterancestors():
                if anc.get("eId", "").startswith("para_"):
                    para_eId = anc.get("eId")
                    break

            out.append({
                "href": href or None,
                "uk_type": uk_type,
                "canonical": canonical,
                "text": txt,
                "para_eId": para_eId,
            })
    return out

def build_citation_map(repo_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      cases_df: nodes
      citations_df: edge occurrences
      edges_df: aggregated edges
    """
    case_dirs = iter_case_dirs(repo_root)

    cases_rows = []
    citation_rows = []

    for case_dir in case_dirs:
        meta = load_meta(case_dir)
        root = parse_xml(case_dir)
        ident = extract_case_identity(root)

        case_id = ident["tna_page"] or meta.get("links", {}).get("html") or meta.get("document_uri")

        # --- Cases table row ---
        cases_rows.append({
            "case_id": case_id,
            "court": meta.get("document_uri", "").split("/")[0] if meta.get("document_uri") else None,
            "document_uri": meta.get("document_uri"),
            "title": meta.get("title"),
            "neutral_citation": ident.get("neutral_citation"),
            "work_name": ident.get("work_name"),
            "tna_page": ident.get("tna_page"),
            "tna_uri": ident.get("tna_uri"),
            "html_link": meta.get("links", {}).get("html"),
            "xml_link": meta.get("links", {}).get("xml"),
            "updated": meta.get("updated"),
            "published": meta.get("published"),
            "local_dir": case_dir,
        })

        # --- paragraph text map for context ---
        para_map = extract_paragraph_index(root)

        # --- citation extraction ---
        refs = extract_case_refs(root)
        for ref in refs:
            para_eId = ref["para_eId"]
            num_txt, num_int, para_text = para_map.get(para_eId, (None, None, None))

            # build a small snippet for provenance
            snippet = None
            if para_text:
                snippet = para_text[:400]

            citation_rows.append({
                "source_case_id": case_id,
                "source_document_uri": meta.get("document_uri"),
                "source_neutral_citation": ident.get("neutral_citation"),

                "target_href": ref.get("href"),
                "target_neutral_citation": ref.get("canonical"),
                "target_text": ref.get("text"),
                "uk_type": ref.get("uk_type"),

                "source_paragraph_eId": para_eId,
                "source_paragraph_num": num_int,
                "source_paragraph_label": num_txt,
                "context_snippet": snippet,
            })

    cases_df = pd.DataFrame(cases_rows).drop_duplicates(subset=["case_id"])
    citations_df = pd.DataFrame(citation_rows)

    # Build aggregated edge list
    # Prefer linking by href; fallback to neutral citation
    citations_df["target_key"] = citations_df["target_href"].fillna(citations_df["target_neutral_citation"])
    edges_df = (
        citations_df
        .dropna(subset=["target_key"])
        .groupby(["source_case_id", "target_key"], as_index=False)
        .agg(
            count_total=("target_key", "size"),
            paras_count=("source_paragraph_eId", pd.Series.nunique),
            first_para=("source_paragraph_num", "min"),
            last_para=("source_paragraph_num", "max"),
        )
    )

    return cases_df, citations_df, edges_df

if __name__ == "__main__":

    cases_df, citations_df, edges_df = build_citation_map("./repo")

    print(cases_df.head())
    print(citations_df.head())
    print(edges_df.sort_values("count_total", ascending=False).head(20))

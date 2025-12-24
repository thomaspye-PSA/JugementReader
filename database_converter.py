import os
import re
import json
import hashlib
from typing import Tuple, Optional, List, Dict, Any
import pandas as pd
from lxml import etree


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _parse_akn_header_fields(xml_path: str) -> Dict[str, Any]:
    """
    Best-effort extraction from Akoma Ntoso data.xml:
      - neutral citation
      - parties
      - judges
      - court/year/number (if present)
      - judgment date
    """
    out = {
        "neutral_citation": None,
        "party_appellant": None,
        "party_respondent": None,
        "judge_list": None,
        "judgment_date": None,
        "court": None,
        "year": None,
        "number": None,
    }

    if not os.path.exists(xml_path):
        return out

    try:
        parser = etree.XMLParser(recover=True, huge_tree=True)
        root = etree.parse(xml_path, parser).getroot()

        # Neutral citation appears often as <neutralCitation> in header
        neutral = root.xpath("string(.//*[local-name()='neutralCitation'][1])")
        if neutral:
            out["neutral_citation"] = neutral.strip() or None

        # UK proprietary fields: <uk:court>, <uk:year>, <uk:number>
        # Use local-name() to ignore namespace prefix issues.
        court = root.xpath("string(.//*[local-name()='court'][1])")
        year = root.xpath("string(.//*[local-name()='year'][1])")
        number = root.xpath("string(.//*[local-name()='number'][1])")

        out["court"] = court.strip().lower() if court and court.strip() else None
        out["year"] = int(year.strip()) if year and year.strip().isdigit() else None
        out["number"] = int(number.strip()) if number and number.strip().isdigit() else None

        # Judgment date often in <docDate date="...">
        doc_date_attr = root.xpath("string(.//*[local-name()='docDate'][1]/@date)")
        if doc_date_attr:
            out["judgment_date"] = doc_date_attr.strip() or None

        # Parties: often <party> in header
        parties = root.xpath(".//*[local-name()='header']//*[local-name()='party']")
        if parties:
            # Heuristic: first is appellant, second is respondent
            # (UKSC judgments follow this often)
            ptexts = []
            for p in parties:
                t = " ".join(p.xpath(".//text()")).strip()
                if t:
                    ptexts.append(t)
            if len(ptexts) >= 1:
                out["party_appellant"] = ptexts[0]
            if len(ptexts) >= 2:
                out["party_respondent"] = ptexts[1]

        # Judges: <judge> tags in header
        judges = root.xpath(".//*[local-name()='header']//*[local-name()='judge']")
        jnames = []
        for j in judges:
            t = " ".join(j.xpath(".//text()")).strip()
            if t:
                jnames.append(t)
        if jnames:
            out["judge_list"] = jnames

    except Exception:
        # Keep best-effort defaults
        pass

    return out


def repo_to_dataframes(
    base_dir: str,
    court: str = "uksc",
    include_paragraphs: bool = False,
    max_cases: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Reads your local repo folder structure:

      {base_dir}/{court}/{document_uri_slug}/
        - meta.json
        - text.txt
        - data.xml

    Returns:
      cases_df: one row per judgment
      paragraphs_df: optional one row per paragraph (if include_paragraphs=True)
    """

    court_dir = os.path.join(base_dir, court.lower())
    if not os.path.exists(court_dir):
        raise FileNotFoundError(f"Could not find court directory: {court_dir}")

    case_rows: List[Dict[str, Any]] = []
    para_rows: List[Dict[str, Any]] = []

    subdirs = [os.path.join(court_dir, d) for d in os.listdir(court_dir)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    subdirs.sort()

    if max_cases is not None:
        subdirs = subdirs[:max_cases]

    for case_path in subdirs:
        meta_path = os.path.join(case_path, "meta.json")
        txt_path = os.path.join(case_path, "text.txt")
        xml_path = os.path.join(case_path, "data.xml")
        paras_path = os.path.join(case_path, "paragraphs.jsonl")

        meta = _load_json(meta_path) or {}
        raw_text = _load_text(txt_path) or ""

        # Extract basic fields from meta.json
        document_uri = meta.get("document_uri")
        title = meta.get("title")
        updated = meta.get("updated")
        published = meta.get("published")
        content_hash = meta.get("content_hash")
        links = meta.get("links") or {}

        # Extract richer header fields from XML
        xml_fields = _parse_akn_header_fields(xml_path)

        # Pick the best neutral citation from XML first, fallback to uk:cite if you stored it later
        neutral_citation = xml_fields.get("neutral_citation")
        if not neutral_citation:
            # Sometimes meta/title includes it, but this is just a fallback
            neutral_citation = None

        # Determine stable case_id
        # Prefer the actual canonical uri for stable IDs
        case_id = document_uri or os.path.basename(case_path)

        # Some sanity fields
        word_count = len(raw_text.split()) if raw_text else 0
        char_count = len(raw_text) if raw_text else 0
        text_sha256 = _sha256_text(raw_text) if raw_text else None

        row = {
            "case_id": case_id,
            "document_uri": document_uri,
            "title": title,
            "court": xml_fields.get("court") or court.lower(),
            "year": xml_fields.get("year"),
            "number": xml_fields.get("number"),
            "neutral_citation": neutral_citation,
            "judgment_date": xml_fields.get("judgment_date"),
            "party_appellant": xml_fields.get("party_appellant"),
            "party_respondent": xml_fields.get("party_respondent"),
            "judge_list": xml_fields.get("judge_list"),
            "updated": updated,
            "published": published,
            "content_hash": content_hash,
            "xml_url": links.get("xml"),
            "pdf_url": links.get("pdf"),
            "html_url": links.get("html"),
            "data_xml_url": links.get("data_xml"),
            "text": raw_text,
            "text_sha256": text_sha256,
            "word_count": word_count,
            "char_count": char_count,
            "source_path": case_path,
        }
        case_rows.append(row)

        # Optional: build paragraph table
        if include_paragraphs:
            # paragraphs = meta.get("paragraphs") or []
            with open(paras_path, "r", encoding="utf-8") as f:
                paragraphs = [json.loads(line) for line in f if line.strip()]

            for p in paragraphs:
                para_number = p.get("num")
                # Clean para_number to be an integer if possible
                para_number = re.sub(r"[^\d]*", "", para_number) if para_number else para_number
                para_number = int(para_number)

                para_text = p.get("text", "")
                para_rows.append(
                    {
                        "case_id": case_id,
                        "eId": p.get("eId"),
                        "para_number": para_number,
                        "text": para_text,
                        "text_len": len(para_text) if para_text else 0,
                        "num_words": len(para_text.split()) if para_text else 0,
                    }
                )

    cases_df = pd.DataFrame(case_rows)

    # Normalize judge_list to JSON strings (so it stores cleanly in SQL later)
    if "judge_list" in cases_df.columns:
        cases_df["judge_list"] = cases_df["judge_list"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else None
        )

    paragraphs_df = pd.DataFrame(para_rows) if include_paragraphs else None
    return cases_df, paragraphs_df


if __name__ == "__main__":

    cases_df, paragraphs_df = repo_to_dataframes(
        base_dir=r".\repo",
        court="uksc",
        include_paragraphs=True
    )

    print(cases_df.shape)
    print(cases_df[["case_id", "neutral_citation", "title", "judgment_date"]].head())

    if paragraphs_df is not None:
        print(paragraphs_df.shape)
        print(paragraphs_df.head())

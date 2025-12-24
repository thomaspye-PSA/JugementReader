import os
import re
import json
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, Any, List
from urllib.parse import urlencode, urlparse

import requests
from lxml import etree
from tqdm import tqdm

BASE = "https://caselaw.nationalarchives.gov.uk"
ATOM_URL = f"{BASE}/atom.xml"

# --- Simple configuration ---
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10240"
)
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.3  # keep well under rate limits (1000 req / 5 min)

# Atom namespaces
NS_ATOM = {
    "atom": "http://www.w3.org/2005/Atom",
    "tna": "https://caselaw.nationalarchives.gov.uk/atom/tna",
}

# Akoma Ntoso namespaces
AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
UK_NS = "https://caselaw.nationalarchives.gov.uk/akn"
HTML_NS = "http://www.w3.org/1999/xhtml"

NS_AKN = {
    "akn": AKN_NS,
    "uk": UK_NS,
    "html": HTML_NS,
}


@dataclass
class AtomEntry:
    uri: str                  # tna:uri (document URI)
    title: str                # atom:title
    updated: str              # atom:updated
    published: Optional[str]  # atom:published (may exist)
    content_hash: Optional[str]  # tna:contenthash (may exist)
    xml_link: Optional[str]   # link rel="alternate" type="application/akn+xml"
    pdf_link: Optional[str]   # link rel="alternate" type="application/pdf"
    html_link: Optional[str]  # link rel="alternate"


# ----------------------------
# Utilities
# ----------------------------

def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\s/]+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s[:180]


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _ensure_absolute_url(href: str) -> str:
    if not href:
        return href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return BASE + href
    return BASE + "/" + href


def _document_uri_from_xml_link(xml_link: str) -> Optional[str]:
    """
    Convert e.g. https://caselaw.nationalarchives.gov.uk/uksc/2025/41/data.xml
    into 'uksc/2025/41'
    """
    if not xml_link:
        return None
    parsed = urlparse(xml_link)
    path = parsed.path.strip("/")
    if path.endswith("data.xml"):
        path = path[: -len("data.xml")].rstrip("/")
    return path or None


def validate_court_slug(session: requests.Session, court: str) -> bool:
    """
    Quickly validate whether a 'court' slug is accepted by the Atom endpoint.
    We request page=1 and see whether we get a 200 and a well-formed feed.
    """
    try:
        feed = fetch_atom_page(session, page=1, per_page=1, court=court, order="-date")
        # must have a root <feed> in Atom namespace
        return feed.tag.endswith("feed")
    except Exception:
        return False


# ----------------------------
# Atom feed scraping
# ----------------------------

def fetch_atom_page(
    session: requests.Session,
    page: int,
    per_page: int = 50,
    court: str = "uksc",
    order: str = "-date",
) -> etree._Element:
    params = {"court": court, "page": page, "per_page": per_page, "order": order}
    url = f"{ATOM_URL}?{urlencode(params)}"
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return etree.fromstring(resp.content)


def parse_atom_entries(feed_xml: etree._Element) -> List[AtomEntry]:
    entries: List[AtomEntry] = []

    for e in feed_xml.findall("atom:entry", namespaces=NS_ATOM):
        title = (e.findtext("atom:title", namespaces=NS_ATOM) or "").strip()
        updated = (e.findtext("atom:updated", namespaces=NS_ATOM) or "").strip()
        published = e.findtext("atom:published", namespaces=NS_ATOM)
        if published:
            published = published.strip()

        uri = (e.findtext("tna:uri", namespaces=NS_ATOM) or "").strip()
        if not uri:
            # fallback if namespace changes
            uri = (e.xpath("string(.//*[local-name()='uri'][1])") or "").strip()

        if not uri:
            continue

        content_hash = e.findtext("tna:contenthash", namespaces=NS_ATOM)
        if content_hash:
            content_hash = content_hash.strip()

        xml_link = None
        pdf_link = None
        html_link = None

        for link in e.findall("atom:link", namespaces=NS_ATOM):
            rel = link.get("rel")
            href = link.get("href")
            typ = link.get("type")

            if rel == "alternate" and href:
                href = _ensure_absolute_url(href)
                if typ == "application/akn+xml":
                    xml_link = href
                elif typ == "application/pdf":
                    pdf_link = href
                elif typ is None:
                    html_link = href

        entries.append(
            AtomEntry(
                uri=uri,
                title=title,
                updated=updated,
                published=published,
                content_hash=content_hash,
                xml_link=xml_link,
                pdf_link=pdf_link,
                html_link=html_link,
            )
        )

    return entries


def find_next_page(feed_xml: etree._Element) -> Optional[int]:
    for link in feed_xml.findall("atom:link", namespaces=NS_ATOM):
        if link.get("rel") == "next":
            href = link.get("href") or ""
            m = re.search(r"[?&]page=(\d+)", href)
            if m:
                return int(m.group(1))
    return None


def iter_court_entries(
    session: requests.Session,
    court: str,
    start_page: int = 1,
    per_page: int = 50,
    order: str = "-date",
) -> Iterator[AtomEntry]:
    page = start_page
    while True:
        feed = fetch_atom_page(session, page=page, per_page=per_page, court=court, order=order)
        entries = parse_atom_entries(feed)
        for entry in entries:
            yield entry

        next_page = find_next_page(feed)
        if not next_page:
            break
        page = next_page


# ----------------------------
# Fetch and parse LegalDocML
# ----------------------------

def fetch_document_xml(session: requests.Session, entry: AtomEntry) -> bytes:
    """
    Prefer entry.xml_link (application/akn+xml), otherwise construct from document URI.
    """
    if entry.xml_link:
        url = entry.xml_link
    else:
        url = f"{BASE}/{entry.uri}/data.xml"

    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def parse_akn_judgment(xml_bytes: bytes) -> Dict[str, Any]:
    """
    Parse Akoma Ntoso judgment (TNA enriched).
    Returns:
      - doc-level structured metadata
      - paragraphs (official)
      - references (deduped)
      - per-paragraph reference links
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml_bytes, parser=parser)

    # ---- Document-level metadata ----
    doc_uri = root.xpath("string(.//akn:FRBRExpression/akn:FRBRthis/@value)", namespaces=NS_AKN)
    work_uri = root.xpath("string(.//akn:FRBRWork/akn:FRBRthis/@value)", namespaces=NS_AKN)
    title = root.xpath("string(.//akn:FRBRWork/akn:FRBRname/@value)", namespaces=NS_AKN)

    judgment_date = root.xpath(
        "string(.//akn:FRBRExpression/akn:FRBRdate[@name='judgment']/@date)",
        namespaces=NS_AKN,
    )

    court = root.xpath("string(.//uk:court)", namespaces=NS_AKN) or None
    year = root.xpath("string(.//uk:year)", namespaces=NS_AKN) or None
    number = root.xpath("string(.//uk:number)", namespaces=NS_AKN) or None
    neutral_citation = root.xpath("string(.//uk:cite)", namespaces=NS_AKN)
    if not neutral_citation:
        neutral_citation = root.xpath("string(.//akn:neutralCitation)", namespaces=NS_AKN)

    delivered_date_attr = root.xpath("string(.//akn:header//akn:docDate/@date)", namespaces=NS_AKN) or None
    delivered_date_display = _clean_text(root.xpath("string(.//akn:header//akn:docDate)", namespaces=NS_AKN)) or None

    # ---- Parties ----
    parties: List[Dict[str, Any]] = []
    for p in root.xpath(".//akn:header//akn:party", namespaces=NS_AKN):
        parties.append({
            "name": _clean_text(" ".join(p.xpath(".//text()"))),
            "role_ref": p.get("as"),
            "person_ref": p.get("refersTo"),
        })

    # ---- Judges ----
    judges: List[Dict[str, Any]] = []
    for j in root.xpath(".//akn:header//akn:judge", namespaces=NS_AKN):
        judges.append({
            "name": _clean_text(" ".join(j.xpath(".//text()"))),
            "ref": j.get("refersTo"),
        })

    # ---- Full text (baseline) ----
    full_text = _clean_text(" ".join(root.xpath(".//text()")))

    # ---- Paragraphs ----
    paragraphs: List[Dict[str, Any]] = []
    para_ref_links: List[Dict[str, Any]] = []

    # We use local-name to be robust across slightly different AKN serialisations,
    # but still keep namespaces for ref parsing.
    for para in root.xpath(".//akn:judgmentBody//*[local-name()='paragraph'][@eId]", namespaces=NS_AKN):
        eId = para.get("eId")
        if not eId or not eId.startswith("para_"):
            continue

        # Direct number (avoid nested embeddedStructure numbers)
        num = _clean_text("".join(para.xpath("./akn:num[1]//text()", namespaces=NS_AKN)))

        # Take direct content only (avoid embeddedStructure footnote paragraphs)
        content_nodes = para.xpath("./akn:content[1]", namespaces=NS_AKN)
        if content_nodes:
            content_node = content_nodes[0]
            text = _clean_text(" ".join(content_node.xpath(".//text()")))
        else:
            # fallback if structure is odd
            text = _clean_text(" ".join(para.xpath(".//text()")))

        if not text:
            continue

        # --- References within this paragraph (exclude nested embeddedStructure refs if desired) ---
        case_refs = []
        leg_refs = []

        for r in para.xpath(".//akn:ref[@uk:type]", namespaces=NS_AKN):
            r_type = r.get(f"{{{UK_NS}}}type")
            ref_obj = {
                "type": r_type,
                "href": r.get("href"),
                "canonical": r.get(f"{{{UK_NS}}}canonical"),
                "is_neutral": r.get(f"{{{UK_NS}}}isNeutral"),
                "origin": r.get(f"{{{UK_NS}}}origin"),
                "text": _clean_text(" ".join(r.xpath(".//text()"))),
            }

            if r_type == "case":
                case_refs.append(ref_obj)
            elif r_type == "legislation":
                leg_refs.append(ref_obj)

        paragraphs.append({
            "eId": eId,
            "num": num,
            "text": text,
            "case_ref_count": len(case_refs),
            "legislation_ref_count": len(leg_refs),
        })

        if case_refs or leg_refs:
            para_ref_links.append({
                "eId": eId,
                "num": num,
                "case_refs": case_refs,
                "legislation_refs": leg_refs,
            })

    # ---- Global references (deduped) ----
    all_refs = []
    for r in root.xpath(".//akn:ref[@uk:type]", namespaces=NS_AKN):
        all_refs.append({
            "type": r.get(f"{{{UK_NS}}}type"),
            "href": r.get("href"),
            "canonical": r.get(f"{{{UK_NS}}}canonical"),
            "is_neutral": r.get(f"{{{UK_NS}}}isNeutral"),
            "origin": r.get(f"{{{UK_NS}}}origin"),
            "text": _clean_text(" ".join(r.xpath(".//text()"))),
        })

    seen = set()
    refs_dedup = []
    for ref in all_refs:
        key = (ref["type"], ref["href"], ref["canonical"], ref["text"])
        if key not in seen:
            seen.add(key)
            refs_dedup.append(ref)

    meta = {
        "doc_uri": doc_uri or None,
        "work_uri": work_uri or None,
        "title": title or None,
        "court": court,
        "year": year,
        "number": number,
        "neutral_citation": neutral_citation or None,
        "judgment_date": judgment_date or None,
        "delivered_date": delivered_date_attr,
        "delivered_date_display": delivered_date_display,
        "parties": parties,
        "judges": judges,
        "paragraph_count": len(paragraphs),
        "reference_count": len(refs_dedup),
    }

    return {
        "meta": meta,
        "full_text": full_text,
        "paragraphs": paragraphs,
        "references": refs_dedup,
        "paragraph_reference_links": para_ref_links,
    }


# ----------------------------
# Write repo outputs
# ----------------------------

def write_case_files(
    base_dir: str,
    entry: AtomEntry,
    xml_bytes: bytes,
    parsed: Dict[str, Any],
    court: str,
) -> str:
    """
    Writes:
      - data.xml
      - text.txt
      - meta.json
      - paragraphs.jsonl
      - references.json

    Output directory:
      base_dir/<court>/<uri_slug>/
    """
    out_dir = os.path.join(base_dir, _safe_slug(court), _safe_slug(entry.uri))
    os.makedirs(out_dir, exist_ok=True)

    xml_path = os.path.join(out_dir, "data.xml")
    txt_path = os.path.join(out_dir, "text.txt")
    meta_path = os.path.join(out_dir, "meta.json")
    paras_path = os.path.join(out_dir, "paragraphs.jsonl")
    refs_path = os.path.join(out_dir, "references.json")

    with open(xml_path, "wb") as f:
        f.write(xml_bytes)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(parsed.get("full_text") or "")

    meta = {
        "document_uri": entry.uri,
        "atom_title": entry.title,
        "updated": entry.updated,
        "published": entry.published,
        "content_hash": entry.content_hash,
        "links": {
            "xml": entry.xml_link,
            "pdf": entry.pdf_link,
            "html": entry.html_link,
            "data_xml": entry.xml_link or f"{BASE}/{entry.uri}/data.xml",
        },
        "akn": parsed["meta"],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(paras_path, "w", encoding="utf-8") as f:
        for p in parsed["paragraphs"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    refs_obj = {
        "references": parsed["references"],
        "paragraph_reference_links": parsed["paragraph_reference_links"],
    }
    with open(refs_path, "w", encoding="utf-8") as f:
        json.dump(refs_obj, f, ensure_ascii=False, indent=2)

    return out_dir


def already_downloaded(base_dir: str, court: str, document_uri: str, content_hash: Optional[str]) -> bool:
    """
    Skip if we already have this document and, if present, the content hash matches.
    """
    out_dir = os.path.join(base_dir, _safe_slug(court), _safe_slug(document_uri))
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        return False

    if not content_hash:
        return True

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("content_hash") == content_hash
    except Exception:
        return False


# ----------------------------
# Main pipeline
# ----------------------------

def build_court_repo(
    base_dir: str,
    court: str,
    max_docs: Optional[int] = None,
    start_page: int = 1,
    per_page: int = 50,
    order: str = "-date",
):
    """
    Generic pipeline for any valid TNA court slug:
    - Iterate Atom feed for <court>
    - Download data.xml for each doc
    - Parse Akoma Ntoso
    - Save raw + structured outputs under base_dir/<court>/
    """
    os.makedirs(base_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if not validate_court_slug(session, court):
        raise ValueError(
            f"Invalid or unsupported court slug for Atom feed: '{court}'. "
            f"Try opening {ATOM_URL}?court=<slug>&page=1&per_page=1"
        )

    count = 0
    desc = f"{court} judgments"
    for entry in tqdm(iter_court_entries(session, court=court, start_page=start_page, per_page=per_page, order=order), desc=desc):
        if max_docs is not None and count >= max_docs:
            break

        if already_downloaded(base_dir, court, entry.uri, entry.content_hash):
            # optional: comment out if too noisy
            # print(f"[SKIP] {entry.uri} (already downloaded with same content hash)")
            continue

        try:
            xml_bytes = fetch_document_xml(session, entry)
            parsed = parse_akn_judgment(xml_bytes)
            write_case_files(base_dir, entry, xml_bytes, parsed, court=court)

            count += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except requests.HTTPError as e:
            print(f"[HTTP ERROR] {entry.uri}: {e}")
            time.sleep(2)
        except etree.XMLSyntaxError as e:
            print(f"[XML ERROR] {entry.uri}: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] {entry.uri}: {e}")
            time.sleep(2)

    print(f"Done. Downloaded/updated {count} documents for court={court}.")


if __name__ == "__main__":

    courts = ["uksc", "ukut/lc", ]
    for court in courts:
        build_court_repo(base_dir="./repo2", court=court, max_docs=50, start_page=1)
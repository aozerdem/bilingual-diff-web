import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import html
import re
from difflib import ndiff, SequenceMatcher
from collections import Counter
from itertools import groupby
import io
import os
import zipfile
import xlsxwriter

# ==========================================
# PART 1: CORE PARSING LOGIC
# ==========================================

def parse_tmx(file_content):
    segments = []
    tree = ET.parse(file_content)
    root = tree.getroot()
    ns = {'xml': 'http://www.w3.org/XML/1998/namespace'}
    
    header = root.find("header")
    src_lang = "en"
    if header is not None:
        src_lang = header.attrib.get("srclang", "en").lower()

    for tu in root.iter("tu"):
        source_text = ""
        target_text = ""
        
        tuvs = tu.findall("tuv")
        if len(tuvs) < 2: continue

        for tuv in tuvs:
            lang = tuv.attrib.get(f"{{{ns['xml']}}}lang", "").lower()
            seg = tuv.find("seg")
            text = seg.text if seg is not None else ""
            
            if not text: continue

            if src_lang in lang:
                source_text = text
            else:
                target_text = text
        
        if not source_text or not target_text:
             seg1 = tuvs[0].find("seg")
             seg2 = tuvs[1].find("seg")
             source_text = seg1.text if seg1 is not None else ""
             target_text = seg2.text if seg2 is not None else ""

        if source_text and target_text:
            segments.append({"source": source_text, "target": target_text})
            
    return segments

def parse_excel(file_content):
    df = pd.read_excel(file_content, engine="openpyxl", header=None)
    v1, v2 = [], []
    for _, row in df.iterrows():
        source = str(row[0]) if not pd.isna(row[0]) else ""
        ver1 = str(row[1]) if not pd.isna(row[1]) else ""
        ver2 = str(row[2]) if not pd.isna(row[2]) else ""
        
        if source:
            v1.append({"source": source, "target": ver1})
            v2.append({"source": source, "target": ver2})
    return v1, v2

def parse_xliff(file_content):
    segments = []
    try:
        tree = ET.parse(file_content)
        root = tree.getroot()
        ns_match = re.match(r'\{(.*)\}', root.tag)
        ns = {'ns': ns_match.group(1)} if ns_match else {}
        
        units = root.findall('.//ns:trans-unit', ns) if ns else root.findall('.//trans-unit')
        
        for trans_unit in units:
            source_el = trans_unit.find('ns:source', ns) if ns else trans_unit.find('source')
            target_el = trans_unit.find('ns:target', ns) if ns else trans_unit.find('target')

            source = "".join(source_el.itertext()).strip() if source_el is not None else ""
            target = "".join(target_el.itertext()).strip() if target_el is not None else ""

            if source or target:
                segments.append({"source": source, "target": target})
    except Exception as e:
        st.error(f"Error parsing XLIFF: {e}")
    return segments

def parse_wol_html(file_content, is_original):
    segments = []
    soup = BeautifulSoup(file_content, 'html.parser')
    tables = soup.find_all("table")
    
    target_table = None
    headers = []
    
    for table in tables:
        rows = table.find_all("tr")
        if not rows: continue
        cells = rows[0].find_all(['th', 'td'])
        current_headers = [c.get_text(" ", strip=True).lower() for c in cells]
        
        if "source" in current_headers and "re translation" in current_headers:
            target_table = table
            headers = current_headers
            break
            
    if target_table is None: return []

    try:
        source_idx = headers.index("source")
        re_idx = headers.index("re translation")
        pe_idx = -1
        
        for h in ["pe translation", "ht translation", "mt translation", "translation"]:
            if h in headers:
                pe_idx = headers.index(h)
                break
        
        if pe_idx == -1: return []

    except ValueError: return []

    for row in target_table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) > max(source_idx, pe_idx, re_idx):
            source = html.unescape(cells[source_idx].get_text(" ", strip=True))
            v1 = html.unescape(cells[pe_idx].get_text(" ", strip=True))
            v2 = html.unescape(cells[re_idx].get_text(" ", strip=True))
            
            segments.append({"source": source, "target": v1 if is_original else v2})
            
    return segments

def load_segments(uploaded_file, is_original=True):
    uploaded_file.seek(0)
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".tmx"):
        return parse_tmx(uploaded_file)
    elif filename.endswith(".xlsx"):
        return parse_excel(uploaded_file)[0] if is_original else parse_excel(uploaded_file)[1]
    elif filename.endswith(".htm") or filename.endswith(".html"):
        return parse_wol_html(uploaded_file, is_original)
    elif filename.endswith((".mxliff", ".sdlxliff", ".xlf", ".xliff")):
        return parse_xliff(uploaded_file)
    else:
        return []

BILINGUAL_EXTS = (".tmx", ".mxliff", ".sdlxliff", ".xlf", ".xliff")

class BilingualFile(io.BytesIO):
    """In-memory bilingual file (uploaded directly or extracted from a zip)."""
    def __init__(self, data, name, display_name):
        super().__init__(data)
        self.name = name                  # base file name, used for parsing and name matching
        self.display_name = display_name  # full path incl. zip name, shown in the UI
        self.lang = detect_target_language(data, display_name)

def detect_target_language(data, path=""):
    """Reads the target language from the file header; falls back to a locale code (e.g. ar-AE) in the path."""
    head = data[:100000].decode("utf-8", errors="ignore")
    m = re.search(r'\b(?:target-language|trgLang)\s*=\s*["\']([^"\']+)["\']', head)
    if m:
        return m.group(1).strip().lower().replace("_", "-")
    if "<tmx" in head:
        src = re.search(r'\bsrclang\s*=\s*["\']([^"\']+)["\']', head)
        src_lang = src.group(1).lower() if src else ""
        for lang in re.findall(r'xml:lang\s*=\s*["\']([^"\']+)["\']', head):
            lang = lang.lower().replace("_", "-")
            if lang != src_lang.replace("_", "-"):
                return lang
    m = re.search(r'(?<![A-Za-z])([a-z]{2,3}[-_][A-Z]{2})(?![A-Za-z])', path)
    if m:
        return m.group(1).lower().replace("_", "-")
    return "unknown"

# Zip-based containers: plain zips and Trados project/return packages
ARCHIVE_EXTS = (".zip", ".sdlppx", ".sdlrpx")

def extract_bilingual_from_archive(data, archive_name, files, skipped, depth=0):
    """Collects bilingual files from a zip-based archive, including archives nested inside it."""
    if not zipfile.is_zipfile(io.BytesIO(data)):
        skipped.append(f"{archive_name} (not a zip archive)")
        return
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or "__MACOSX" in info.filename: continue
            full_name = f"{archive_name}/{info.filename}"
            lower = info.filename.lower()
            if lower.endswith(BILINGUAL_EXTS):
                files.append(BilingualFile(zf.read(info), os.path.basename(info.filename), full_name))
            elif lower.endswith(ARCHIVE_EXTS) and depth < 3:
                extract_bilingual_from_archive(zf.read(info), full_name, files, skipped, depth + 1)
            else:
                skipped.append(full_name)

def collect_bilingual_files(uploaded_files):
    """Expands archive uploads and keeps only bilingual files. Returns (files, skipped_names)."""
    files, skipped = [], []
    for up in uploaded_files or []:
        if up.name.lower().endswith(ARCHIVE_EXTS):
            extract_bilingual_from_archive(up.getvalue(), up.name, files, skipped)
        elif up.name.lower().endswith(BILINGUAL_EXTS):
            files.append(BilingualFile(up.getvalue(), up.name, up.name))
        else:
            skipped.append(up.name)
    return files, skipped

# ==========================================
# PART 2: EXPORT & COMPARE LOGIC
# ==========================================

def tokenize_for_ter(text: str) -> list:
    """Industry-standard tokenizer for TER: lowercases and isolates punctuation symbols."""
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

def get_valid_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    valid = []
    for w in words:
        if w.isdigit():
            valid.append(w)
        elif w.isalpha():
            if len(w) >= 3:
                valid.append(w)
        else:
             valid.append(w)
    return valid

def highlight_differences(v1, v2):
    diff = list(ndiff(v1, v2))
    highlighted_v1 = ''
    highlighted_v2 = ''
    for d in diff:
        code = d[0]
        char = d[2:]
        if code == ' ':
            highlighted_v1 += char
            highlighted_v2 += char
        elif code == '-':
            highlighted_v1 += f'<span style="color:red; background-color: #ffdce0;">{char}</span>'
        elif code == '+':
            highlighted_v2 += f'<span style="color:green; background-color: #e2ffdc;">{char}</span>'
    return highlighted_v1, highlighted_v2

def generate_output_filename(mode, v1_file, v2_file=None):
    if mode == "Bilingual Files (TMX/XLIFF)" and v1_file and v2_file:
        name1 = v1_file.name
        name2 = v2_file.name
        lang_code = getattr(v1_file, "lang", None)
        if not lang_code or lang_code == "unknown":
            lang_match = re.search(r'([a-z]{2}[-_][a-z]{2})', name1, re.IGNORECASE)
            lang_code = lang_match.group(1).lower() if lang_match else "unknown"
        part1 = name1[:15]
        part2 = name2[:15]
        return f"{lang_code}_{part1}_vs_{part2}.html"
    elif mode in ["Excel (3 Columns)", "WOL Report"] and v1_file:
        base_name = os.path.splitext(v1_file.name)[0]
        return f"{base_name}_Report.html"
    return "BilingualDiff_Report.html"

def calculate_word_edits(original: str, edited: str) -> int:
    """Calculates normalized, word-level Levenshtein distance."""
    words1 = tokenize_for_ter(original)
    words2 = tokenize_for_ter(edited)
    n, m = len(words1), len(words2)
    if n == 0: return m
    if m == 0: return n
    matrix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): matrix[i][0] = i
    for j in range(m + 1): matrix[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if words1[i - 1] == words2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1, 
                matrix[i][j - 1] + 1, 
                matrix[i - 1][j - 1] + cost
            )
    return matrix[n][m]

ATTRIBUTION = "BilingualDiff v.1.2 - Developed by Ahmet Ozerdem"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

def empty_totals():
    return {"total": 0, "changed": 0, "len_v1": 0, "len_v2": 0, "word_edits": 0, "ref_words": 0,
            "changed_scores": [], "removed": Counter(), "added": Counter()}

def compare_segments(v1_segs, v2_segs):
    """Compares segments by position. Returns (rows, totals); totals are raw sums so comparisons can be merged."""
    rows = []
    totals = empty_totals()
    totals["total"] = len(v1_segs)

    for i, (seg1, seg2) in enumerate(zip(v1_segs, v2_segs), 1):
        source = seg1.get("source", "")
        v1 = seg1.get("target", "")
        v2 = seg2.get("target", "")

        totals["len_v1"] += len(v1)
        totals["len_v2"] += len(v2)

        # Track words for Global TER using standard normalization token count
        v1_word_count = len(tokenize_for_ter(v1))
        totals["ref_words"] += v1_word_count

        status = "Same"
        score = 100
        ter_score = 0.0

        if v1 != v2:
            status = "Different"
            totals["changed"] += 1

            # Similarity
            score = round(SequenceMatcher(None, v1, v2).ratio() * 100, 1)
            if score < 100: totals["changed_scores"].append(score)

            # Normalized TER Calculation
            edits = calculate_word_edits(v1, v2)
            totals["word_edits"] += edits
            if v1_word_count > 0:
                ter_score = round((edits / v1_word_count) * 100, 1)
            else:
                ter_score = 100.0 if len(tokenize_for_ter(v2)) > 0 else 0.0

            # Words
            w1 = Counter(get_valid_words(v1))
            w2 = Counter(get_valid_words(v2))
            totals["removed"].update((w1 - w2).elements())
            totals["added"].update((w2 - w1).elements())

        rows.append((i, source, v1, v2, status, score, ter_score))

    return rows, totals

def merge_totals(totals_list):
    merged = empty_totals()
    for totals in totals_list:
        for key, value in totals.items():
            merged[key] += value
    return merged

def summarize_totals(totals):
    """Turns raw totals into the report statistics."""
    changed_scores = totals["changed_scores"]
    edit_categories = {"Minor Edit (>85%)": 0, "Medium Edit (50-85%)": 0, "Major Rewrite (<50%)": 0}
    for s in changed_scores:
        if s > 85: edit_categories["Minor Edit (>85%)"] += 1
        elif s >= 50: edit_categories["Medium Edit (50-85%)"] += 1
        else: edit_categories["Major Rewrite (<50%)"] += 1

    total, changed, len_v1 = totals["total"], totals["changed"], totals["len_v1"]
    return {
        "total": total,
        "changed": changed,
        "pct": (changed / total * 100) if total > 0 else 0,
        "expansion": ((totals["len_v2"] - len_v1) / len_v1 * 100) if len_v1 > 0 else 0,
        "avg_score": sum(changed_scores) / len(changed_scores) if changed_scores else 100,
        "corpus_ter": (totals["word_edits"] / totals["ref_words"] * 100) if totals["ref_words"] > 0 else 0,
        "top_removed": totals["removed"].most_common(5),
        "top_added": totals["added"].most_common(5),
        "graph_data": edit_categories
    }

def filter_rows(rows, filter_option):
    if filter_option == "diff": return [r for r in rows if r[4] == "Different"]
    if filter_option == "same": return [r for r in rows if r[4] == "Same"]
    return rows

def format_top_words(lst):
    return ", ".join([f"{w} ({c})" for w, c in lst]) if lst else "None"

def html_page(title, body):
    return f"""
    <html><head><meta charset="UTF-8"><title>{title}</title>
    <style>
        body {{ font-family: Segoe UI, sans-serif; padding: 20px; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
        th {{ background-color: #f2f2f2; text-align: left; }}
        .diff {{ background-color: #fff9db; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 20px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .stat-val {{ font-size: 1.2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; color: #666; }}
        .attribution {{ font-size: 0.8em; color: #888; text-align: right; margin-bottom: 4px; }}
        tr.subtotal td {{ font-weight: bold; background-color: #f2f2f2; }}
        tr.total td {{ font-weight: bold; background-color: #e0e0e0; }}
        h3 {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd; }}
    </style>
    </head><body>
    <div class="attribution">{ATTRIBUTION}</div>
    {body}
    </body></html>"""

def render_html_stats(stats):
    return f"""
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-val">{stats['total']}</div><div class="stat-label">Total Strings</div></div>
        <div class="stat-card"><div class="stat-val">{stats['changed']} ({stats['pct']:.1f}%)</div><div class="stat-label">Changed Strings</div></div>
        <div class="stat-card"><div class="stat-val">{stats['corpus_ter']:.1f}%</div><div class="stat-label">Corpus TER</div></div>
        <div class="stat-card"><div class="stat-val">{stats['avg_score']:.1f}%</div><div class="stat-label">Avg Similarity (on edits)</div></div>
        <div class="stat-card"><div class="stat-val">{stats['expansion']:+.1f}%</div><div class="stat-label">Expansion Factor</div></div>
    </div>

    <p><b>Top Removed:</b> {format_top_words(stats['top_removed'])}<br>
    <b>Top Added:</b> {format_top_words(stats['top_added'])}</p>
    """

def render_html_table(rows, filter_option):
    html_out = """
    <table>
    <tr><th>ID</th><th>Source</th><th>Original Version</th><th>Updated Version</th><th>Sim %</th><th>TER %</th></tr>
    """
    for seg_id, src, v1, v2, status, score, ter_score in filter_rows(rows, filter_option):
        v1_hl, v2_hl = highlight_differences(v1, v2)
        row_class = "diff" if status == "Different" else ""
        ter_str = f"{ter_score:.1f}%" if status == "Different" else "-"
        html_out += f'<tr class="{row_class}"><td>{seg_id}</td><td>{src}</td><td>{v1_hl}</td><td>{v2_hl}</td><td>{score}%</td><td>{ter_str}</td></tr>'
    return html_out + "</table>"

def generate_html_report(rows, stats, filter_option):
    body = "<h2>Comparison Report</h2>" + render_html_stats(stats) + render_html_table(rows, filter_option)
    return html_page("BilingualDiff Report", body)

def add_report_formats(workbook):
    return {
        'header': workbook.add_format({'bold': True, 'bg_color': '#f2f2f2', 'border': 1}),
        'cell': workbook.add_format({'border': 1, 'text_wrap': True, 'valign': 'top'}),
        'diff_bg': workbook.add_format({'bg_color': '#fff9db', 'border': 1, 'text_wrap': True, 'valign': 'top'}),
        'red': workbook.add_format({'font_color': '#9c0006', 'bg_color': '#ffdce0', 'font_strikeout': True}),
        'green': workbook.add_format({'font_color': '#006100', 'bg_color': '#e2ffdc'}),
        'default': workbook.add_format({'font_color': '#000000'}),
        'title': workbook.add_format({'bold': True, 'font_size': 16}),
        'subtitle': workbook.add_format({'italic': True, 'font_color': '#555555'}),
        'attr': workbook.add_format({'italic': True, 'font_color': '#888888', 'align': 'right'}),
        'stat_lbl': workbook.add_format({'font_color': '#666666'}),
        'stat_val': workbook.add_format({'bold': True, 'font_size': 14}),
        'top_words': workbook.add_format({'text_wrap': True, 'valign': 'top'}),
        'link': workbook.add_format({'font_color': 'blue', 'underline': 1}),
    }

def optimize_chunks(chunks):
    """Joins neighbouring [format, text] chunks that share a format."""
    opt = []
    curr_fmt, curr_str = None, ""
    for j in range(0, len(chunks), 2):
        fmt, text = chunks[j], chunks[j+1]
        if fmt == curr_fmt: curr_str += text
        else:
            if curr_str: opt.extend([curr_fmt, curr_str])
            curr_fmt, curr_str = fmt, text
    if curr_str: opt.extend([curr_fmt, curr_str])
    return opt

REPORT_TABLE_START_ROW = 8

def write_dashboard(worksheet, fmt, stats, title, last_col):
    """Writes the title and statistics block (rows 0-6) at the top of a report sheet."""
    worksheet.write(0, 0, title, fmt['title'])
    worksheet.merge_range(0, last_col - 1, 0, last_col, ATTRIBUTION, fmt['attr'])

    labels = ["Total Strings", "Changed Strings", "Corpus TER", "Avg Sim (on edits)", "Expansion Factor"]
    values = [stats['total'], f"{stats['changed']} ({stats['pct']:.1f}%)", f"{stats['corpus_ter']:.1f}%",
              f"{stats['avg_score']:.1f}%", f"{stats['expansion']:+.1f}%"]
    for col, (label, value) in enumerate(zip(labels, values)):
        worksheet.write(2, col, label, fmt['stat_lbl'])
        worksheet.write(3, col, value, fmt['stat_val'])

    worksheet.merge_range(5, 0, 5, last_col, f"Top Removed: {format_top_words(stats['top_removed'])}", fmt['top_words'])
    worksheet.merge_range(6, 0, 6, last_col, f"Top Added: {format_top_words(stats['top_added'])}", fmt['top_words'])

def write_comparison_sheet(worksheet, fmt, rows, stats, filter_option):
    """Writes one comparison (dashboard + colour-coded segment table) to a worksheet. Row 1 is left free for a subtitle."""
    write_dashboard(worksheet, fmt, stats, "Comparison Report", last_col=5)

    headers = ['ID', 'Source', 'Original Version', 'Updated Version', 'Sim %', 'TER %']
    for col_num, header in enumerate(headers):
        worksheet.write(REPORT_TABLE_START_ROW, col_num, header, fmt['header'])

    worksheet.set_column('A:A', 6)
    worksheet.set_column('B:D', 45)
    worksheet.set_column('E:F', 10)

    row_idx = REPORT_TABLE_START_ROW + 1

    for i, source, v1, v2, status, score, ter_score in filter_rows(rows, filter_option):
        current_fmt = fmt['diff_bg'] if status == "Different" else fmt['cell']

        worksheet.write(row_idx, 0, i, current_fmt)
        worksheet.write(row_idx, 1, source, current_fmt)
        worksheet.write(row_idx, 4, f"{score}%", current_fmt)
        worksheet.write(row_idx, 5, f"{ter_score:.1f}%" if status == "Different" else "-", current_fmt)

        if status == "Different":
            v1_chunks, v2_chunks = [], []

            for d in ndiff(v1, v2):
                code, char = d[0], d[2:]
                if code == ' ':
                    v1_chunks.extend([fmt['default'], char])
                    v2_chunks.extend([fmt['default'], char])
                elif code == '-':
                    v1_chunks.extend([fmt['red'], char])
                elif code == '+':
                    v2_chunks.extend([fmt['green'], char])

            v1_opt = optimize_chunks(v1_chunks)
            v2_opt = optimize_chunks(v2_chunks)

            if len(v1_opt) > 2: worksheet.write_rich_string(row_idx, 2, *v1_opt, current_fmt)
            else: worksheet.write(row_idx, 2, v1_opt[1] if len(v1_opt)==2 else v1, current_fmt)

            if len(v2_opt) > 2: worksheet.write_rich_string(row_idx, 3, *v2_opt, current_fmt)
            else: worksheet.write(row_idx, 3, v2_opt[1] if len(v2_opt)==2 else v2, current_fmt)
        else:
            worksheet.write(row_idx, 2, v1, current_fmt)
            worksheet.write(row_idx, 3, v2, current_fmt)

        row_idx += 1

def generate_excel_report(rows, stats, filter_option):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    write_comparison_sheet(workbook.add_worksheet('Comparison Report'), add_report_formats(workbook), rows, stats, filter_option)
    workbook.close()
    return output.getvalue()

def name_similarity(f1, f2):
    """(file name score, full path score) - the path score only breaks ties between identical names."""
    n1, n2 = os.path.splitext(f1.name)[0].lower(), os.path.splitext(f2.name)[0].lower()
    p1 = getattr(f1, "display_name", f1.name).lower()
    p2 = getattr(f2, "display_name", f2.name).lower()
    return SequenceMatcher(None, n1, n2).ratio(), SequenceMatcher(None, p1, p2).ratio()

def match_files_by_name(orig_files, upd_files, threshold):
    """Pairs original/updated files one-to-one by file name similarity (best matches first).
    Files are only paired when their target languages are the same."""
    if len(orig_files) == 1 and len(upd_files) == 1:
        o, u = orig_files[0], upd_files[0]
        return [(o, u, name_similarity(o, u)[0])], [], []

    candidates = []
    for i, o in enumerate(orig_files):
        for j, u in enumerate(upd_files):
            if getattr(o, "lang", None) != getattr(u, "lang", None): continue
            candidates.append((*name_similarity(o, u), i, j))
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    used_o, used_u, pairs = set(), set(), []
    for score, _, i, j in candidates:
        if score < threshold: break
        if i in used_o or j in used_u: continue
        used_o.add(i)
        used_u.add(j)
        pairs.append((orig_files[i], upd_files[j], score))

    unmatched_o = [f for i, f in enumerate(orig_files) if i not in used_o]
    unmatched_u = [f for j, f in enumerate(upd_files) if j not in used_u]
    return pairs, unmatched_o, unmatched_u

def make_unique_filename(filename, taken):
    base, ext = os.path.splitext(filename)
    candidate, n = filename, 2
    while candidate in taken:
        candidate = f"{base}_{n}{ext}"
        n += 1
    taken.add(candidate)
    return candidate

def generate_report_based_on_toggle(comparison, filter_val, export_fmt, mode, f1, f2=None):
    """comparison: (rows, totals) from compare_segments."""
    rows, totals = comparison
    stats = summarize_totals(totals)
    if export_fmt == "HTML Report":
        data = generate_html_report(rows, stats, filter_val)
        filename = generate_output_filename(mode, f1, f2)
        mime = "text/html"
    else:
        data = generate_excel_report(rows, stats, filter_val)
        filename = generate_output_filename(mode, f1, f2).replace(".html", ".xlsx")
        mime = XLSX_MIME
    return data, stats, filename, mime

# --- Consolidated report (several file pairs in one file) ---
# items: one dict per pair with lang, original_name, updated_name, rows, totals and stats, sorted by lang.

def pair_count(n):
    return f"{n} pair" if n == 1 else f"{n} pairs"

def consolidated_summary(items):
    """Summary lines: each pair, a subtotal after each language (when there are several), then the grand total."""
    groups = [(lang, list(group)) for lang, group in groupby(enumerate(items, 1), key=lambda e: e[1]['lang'])]
    lines = []
    for lang, group in groups:
        for n, item in group:
            lines.append({"kind": "pair", "n": n, "lang": lang, "original": item['original_name'],
                          "updated": item['updated_name'], "stats": item['stats']})
        if len(groups) > 1:
            lines.append({"kind": "subtotal", "lang": lang, "original": f"Subtotal ({pair_count(len(group))})", "updated": "",
                          "stats": summarize_totals(merge_totals(item['totals'] for _, item in group))})
    lines.append({"kind": "total", "lang": "", "original": f"Grand Total ({pair_count(len(items))})", "updated": "",
                  "stats": summarize_totals(merge_totals(item['totals'] for item in items))})
    return lines

def consolidated_caption(items):
    langs = len({item['lang'] for item in items})
    return f"{pair_count(len(items))} across {langs} target language{'' if langs == 1 else 's'}"

def generate_consolidated_html(items, filter_option):
    lines = consolidated_summary(items)
    parts = [f"<h2>Consolidated Report</h2><p>{consolidated_caption(items)}</p>",
             render_html_stats(lines[-1]['stats']),
             "<table><tr><th>Target Lang</th><th>Original File</th><th>Updated File</th><th>Total Strings</th><th>Changed</th>"
             "<th>Changed %</th><th>Corpus TER %</th><th>Avg Sim %</th><th>Expansion %</th></tr>"]
    for line in lines:
        s = line['stats']
        original = html.escape(line['original'])
        if line['kind'] == "pair":
            original = f'<a href="#pair-{line["n"]}">{original}</a>'
        parts.append(f'<tr class="{line["kind"]}"><td>{html.escape(line["lang"])}</td><td>{original}</td><td>{html.escape(line["updated"])}</td>'
                     f'<td>{s["total"]}</td><td>{s["changed"]}</td><td>{s["pct"]:.1f}%</td><td>{s["corpus_ter"]:.1f}%</td>'
                     f'<td>{s["avg_score"]:.1f}%</td><td>{s["expansion"]:+.1f}%</td></tr>')
    parts.append("</table>")

    for n, item in enumerate(items, 1):
        parts.append(f'<h3 id="pair-{n}">{n}. [{html.escape(item["lang"])}] {html.escape(item["original_name"])} ↔ {html.escape(item["updated_name"])}</h3>'
                     '<p><a href="#top">↑ Back to summary</a></p>')
        parts.append(render_html_stats(item['stats']))
        parts.append(render_html_table(item['rows'], filter_option))
    return html_page("BilingualDiff Consolidated Report", "".join(parts))

def add_summary_formats(workbook):
    """Summary table formats per line kind (pair / subtotal / total) and cell type."""
    line_styles = {"pair": {}, "subtotal": {'bold': True, 'bg_color': '#f2f2f2'}, "total": {'bold': True, 'bg_color': '#d9d9d9'}}
    cell_styles = {"text": {'text_wrap': True}, "int": {}, "pct": {'num_format': '0.0%'},
                   "exp": {'num_format': '+0.0%;-0.0%;0.0%'}, "link": {'font_color': 'blue', 'underline': 1}}
    return {kind: {name: workbook.add_format({'border': 1, 'valign': 'top', **line_style, **cell_style})
                   for name, cell_style in cell_styles.items()}
            for kind, line_style in line_styles.items()}

def excel_sheet_name(n, item):
    """Unique sheet name for the n-th pair; Excel allows at most 31 chars and no []:*?/\\ or quotes."""
    base = os.path.splitext(os.path.basename(item['original_name']))[0]
    return re.sub(r"[\[\]:*?/\\']", "_", f"{n:02d}_{item['lang']}_{base}")[:31]

def generate_consolidated_excel(items, filter_option):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    fmt = add_report_formats(workbook)
    summary_fmt = add_summary_formats(workbook)
    sheet_names = [excel_sheet_name(n, item) for n, item in enumerate(items, 1)]
    lines = consolidated_summary(items)

    summary = workbook.add_worksheet('Summary')
    write_dashboard(summary, fmt, lines[-1]['stats'], "Consolidated Report", last_col=9)
    summary.write(1, 0, consolidated_caption(items), fmt['subtitle'])

    headers = ['Target Lang', 'Original File', 'Updated File', 'Total Strings', 'Changed', 'Changed %',
               'Corpus TER %', 'Avg Sim %', 'Expansion %', 'Sheet']
    for col, header in enumerate(headers):
        summary.write(REPORT_TABLE_START_ROW, col, header, fmt['header'])
    summary.set_column('A:A', 14)
    summary.set_column('B:C', 40)
    summary.set_column('D:I', 18)
    summary.set_column('J:J', 34)

    for row, line in enumerate(lines, REPORT_TABLE_START_ROW + 1):
        style, s = summary_fmt[line['kind']], line['stats']
        cells = [(line['lang'], 'text'), (line['original'], 'text'), (line['updated'], 'text'),
                 (s['total'], 'int'), (s['changed'], 'int'), (s['pct'] / 100, 'pct'), (s['corpus_ter'] / 100, 'pct'),
                 (s['avg_score'] / 100, 'pct'), (s['expansion'] / 100, 'exp')]
        for col, (value, kind) in enumerate(cells):
            summary.write(row, col, value, style[kind])
        if line['kind'] == "pair":
            name = sheet_names[line['n'] - 1]
            summary.write_url(row, 9, f"internal:'{name}'!A1", style['link'], name)
        else:
            summary.write_blank(row, 9, None, style['text'])

    for name, item in zip(sheet_names, items):
        sheet = workbook.add_worksheet(name)
        write_comparison_sheet(sheet, fmt, item['rows'], item['stats'], filter_option)
        sheet.write(1, 0, f"[{item['lang']}]  {item['original_name']}  ↔  {item['updated_name']}", fmt['subtitle'])
        sheet.write_url(1, 4, "internal:'Summary'!A1", fmt['link'], "← Back to Summary")

    workbook.close()
    return output.getvalue()

def generate_consolidated_report(items, filter_val, export_fmt):
    """One file for a batch: summary (per pair, per language, overall) followed by each pair's full comparison."""
    if export_fmt == "HTML Report":
        return {"filename": "BilingualDiff_Consolidated_Report.html", "data": generate_consolidated_html(items, filter_val), "mime": "text/html"}
    return {"filename": "BilingualDiff_Consolidated_Report.xlsx", "data": generate_consolidated_excel(items, filter_val), "mime": XLSX_MIME}

# ==========================================
# PART 3: STREAMLIT UI
# ==========================================

st.set_page_config(page_title="BilingualDiff", layout="wide")

st.markdown("""
    <style>
    div[data-testid="stDownloadButton"] button {
        background-color: #2e7d32 !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        padding: 10px 24px !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100%;
        margin-top: 20px !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #1b5e20 !important;
        color: white !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 BilingualDiff (Secure Web Edition)")
st.markdown("Compare translation versions securely. **Files never leave your browser.**")

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Select Mode", ["Bilingual Files (TMX/XLIFF)", "Excel (3 Columns)", "WOL Report"])
    filter_opt = st.radio("Export Filter", ["All Segments", "Only DIFFERENT", "Only SAME"], index=0)
    filter_map = {"All Segments": "all", "Only DIFFERENT": "diff", "Only SAME": "same"}
    
    export_format = st.radio("Export Format", ["HTML Report", "Excel (.xlsx) Report"], index=1)

v1_file = None
v2_file = None

# 1. BILINGUAL FILES MODE
if mode == "Bilingual Files (TMX/XLIFF)":
    if 'excel_results' in st.session_state: del st.session_state['excel_results']
    
    col1, col2 = st.columns(2)
    upload_types = ["tmx", "mxliff", "sdlxliff", "xlf", "xliff", "zip", "sdlppx", "sdlrpx"]
    v1_uploads = col1.file_uploader("Upload Original Version(s)", type=upload_types, accept_multiple_files=True)
    v2_uploads = col2.file_uploader("Upload Updated Version(s)", type=upload_types, accept_multiple_files=True)
    st.caption("Zip files and Trados packages (SDLPPX/SDLRPX) are searched for bilingual files (TMX/XLIFF/MXLIFF/SDLXLIFF); other files are skipped. "
               "Files are only paired with files of the same target language.")
    match_threshold = st.slider("File Name Match Threshold (%)", 50, 100, 90,
                                help="Original and updated files are paired by file name similarity. Pairs below this score are left unmatched.")

    v1_files, v1_skipped = collect_bilingual_files(v1_uploads)
    v2_files, v2_skipped = collect_bilingual_files(v2_uploads)
    if v1_skipped or v2_skipped:
        with st.expander(f"Skipped {len(v1_skipped) + len(v2_skipped)} non-bilingual file(s)"):
            for name in v1_skipped + v2_skipped: st.write(name)

    pairs, unmatched_v1, unmatched_v2 = [], [], []
    if v1_files and v2_files:
        pairs, unmatched_v1, unmatched_v2 = match_files_by_name(v1_files, v2_files, match_threshold / 100)
        st.subheader("🔗 File Mapping")
        if pairs:
            st.dataframe(pd.DataFrame([
                {"Target Lang": o.lang, "Original File": o.display_name, "Updated File": u.display_name, "Name Match %": round(s * 100, 1)}
                for o, u, s in sorted(pairs, key=lambda p: (p[0].lang, p[0].display_name))
            ]), use_container_width=True)
        if unmatched_v1:
            st.warning("Unmatched original files (skipped): " + ", ".join(f"{f.display_name} [{f.lang}]" for f in unmatched_v1))
        if unmatched_v2:
            st.warning("Unmatched updated files (skipped): " + ", ".join(f"{f.display_name} [{f.lang}]" for f in unmatched_v2))
    elif v1_uploads and v2_uploads:
        st.warning("No bilingual files found in the uploads.")

    consolidate = len(pairs) > 1 and st.checkbox(
        "Also create a consolidated report",
        help="One extra file in the selected export format: a summary of all pairs (per pair, per target language and overall) "
             "followed by the full comparison of each pair.")

    # Drop old results when the uploads or settings change
    current_sig = ([f.name for f in v1_uploads or []], [f.name for f in v2_uploads or []], match_threshold, filter_opt, export_format, consolidate)
    if st.session_state.get('bilingual_sig') != current_sig:
        st.session_state.pop('bilingual_results', None)
        st.session_state.pop('bilingual_consolidated', None)

    if st.button("Compare & Generate Report"):
        if not v1_files or not v2_files:
            st.warning("Please upload both original and updated files.")
        elif not pairs:
            st.error("No file pairs could be matched. Try lowering the match threshold.")
        else:
            results_storage = []
            consolidated_items = []
            taken_names = set()
            for o_file, u_file, name_score in sorted(pairs, key=lambda p: (p[0].lang, p[0].display_name)):
                with st.spinner(f"Processing {o_file.display_name}..."):
                    try:
                        v1_segs = load_segments(o_file, True)
                        v2_segs = load_segments(u_file, False)
                        comparison = compare_segments(v1_segs, v2_segs)
                        file_data, stats, out_filename, mime_type = generate_report_based_on_toggle(
                            comparison, filter_map[filter_opt], export_format, mode, o_file, u_file
                        )
                        result = {
                            "filename": make_unique_filename(out_filename, taken_names),
                            "lang": o_file.lang,
                            "original_name": o_file.display_name,
                            "updated_name": u_file.display_name,
                            "seg_counts": (len(v1_segs), len(v2_segs)),
                            "data": file_data,
                            "mime": mime_type,
                            "stats": stats
                        }
                        results_storage.append(result)
                        if consolidate:
                            rows, totals = comparison
                            consolidated_items.append(dict(result, rows=rows, totals=totals))
                    except Exception as e:
                        st.error(f"Error processing {o_file.display_name} / {u_file.display_name}: {e}")

            consolidated_report = None
            if len(consolidated_items) > 1:
                with st.spinner("Building consolidated report..."):
                    try:
                        consolidated_report = generate_consolidated_report(consolidated_items, filter_map[filter_opt], export_format)
                    except Exception as e:
                        st.error(f"Error building the consolidated report: {e}")
            st.session_state['bilingual_results'] = results_storage
            st.session_state['bilingual_consolidated'] = consolidated_report
            st.session_state['bilingual_sig'] = current_sig

    results = st.session_state.get('bilingual_results')
    if results:
        st.success(f"Comparison Complete! Processed {len(results)} file pair(s).")

        for res in results:
            n1, n2 = res['seg_counts']
            if n1 != n2:
                st.warning(f"Segment count mismatch in {res['original_name']} ({n1}) vs {res['updated_name']} ({n2}). "
                           "Segments are compared by position, so results may be misaligned.")

        if len(results) > 1:
            consolidated = st.session_state.get('bilingual_consolidated')
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for res in results:
                    zf.writestr(f"{res['lang']}/{res['filename']}", res['data'] if isinstance(res['data'], bytes) else res['data'].encode('utf-8'))
                if consolidated:
                    zf.writestr(consolidated['filename'], consolidated['data'] if isinstance(consolidated['data'], bytes) else consolidated['data'].encode('utf-8'))
            st.download_button(label="📦 DOWNLOAD ALL REPORTS (.ZIP)", data=zip_buffer.getvalue(),
                               file_name="BilingualDiff_All_Reports.zip", mime="application/zip", key="bi_download_zip")
            if consolidated:
                st.download_button(label=f"📑 DOWNLOAD CONSOLIDATED REPORT ({consolidated['filename']})", data=consolidated['data'],
                                   file_name=consolidated['filename'], mime=consolidated['mime'], key="bi_download_consolidated")

            st.divider()
            st.subheader("📊 Batch Summary")
            st.dataframe(pd.DataFrame([{
                "Target Lang": r['lang'],
                "Original File": r['original_name'],
                "Updated File": r['updated_name'],
                "Total Strings": r['stats']['total'],
                "Changed": f"{r['stats']['changed']} ({r['stats']['pct']:.1f}%)",
                "Corpus TER %": round(r['stats']['corpus_ter'], 1),
                "Avg Sim %": round(r['stats']['avg_score'], 1),
                "Expansion %": round(r['stats']['expansion'], 1),
            } for r in results]), use_container_width=True)
            st.divider()

            for i, res in enumerate(results):
                st.write(f"**{res['original_name']}** ↔ **{res['updated_name']}**")
                st.download_button(label=f"⬇️ DOWNLOAD REPORT ({res['filename']})", data=res['data'],
                                   file_name=res['filename'], mime=res['mime'], key=f"bi_dl_btn_{i}")
                st.markdown("---")
        else:
            res = results[0]
            stats = res['stats']
            st.download_button(label=f"⬇️ DOWNLOAD REPORT ({res['filename']})", data=res['data'],
                               file_name=res['filename'], mime=res['mime'], key="bi_dl_single")

            st.divider()
            st.subheader("📊 Translation Analytics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Strings", stats["total"], help="Total count of segments found.")
            m2.metric("Changed Strings", f"{stats['changed']} ({stats['pct']:.1f}%)", help="Number of segments with changes.")
            m3.metric("Corpus TER", f"{stats['corpus_ter']:.1f}%", help="Translation Edit Rate (Total Word Edits / Total Original Words). Lower is better.")
            m4.metric("Avg Edit Similarity", f"{stats['avg_score']:.1f}%", help="Character-level Levenshtein ratio on changed segments. Higher is better.")
            m5.metric("Expansion Factor", f"{stats['expansion']:+.1f}%", help="Length difference.")
            st.divider()

# 2. EXCEL MULTI-FILE MODE 
elif mode == "Excel (3 Columns)":
    v1_files = st.file_uploader("Upload Excel Files", type=["xlsx"], accept_multiple_files=True)
    st.info("Excel must have Source in Col A, Original in Col B, Updated in Col C.")

    if st.button("Compare & Generate Reports"):
        if not v1_files:
            st.warning("Please upload at least one Excel file.")
        else:
            results_storage = []
            for excel_file in v1_files:
                with st.spinner(f"Processing {excel_file.name}..."):
                    try:
                        v1_segs, v2_segs = parse_excel(excel_file)
                        file_data, stats, out_filename, mime_type = generate_report_based_on_toggle(
                            compare_segments(v1_segs, v2_segs), filter_map[filter_opt], export_format, mode, excel_file
                        )
                        
                        results_storage.append({
                            "filename": out_filename,
                            "original_name": excel_file.name,
                            "html": file_data,
                            "stats": stats
                        })
                    except Exception as e:
                        st.error(f"Error processing {excel_file.name}: {e}")
            
            st.session_state['excel_results'] = results_storage

    if 'excel_results' in st.session_state and st.session_state['excel_results']:
        st.success(f"Processed {len(st.session_state['excel_results'])} files.")
        
        if len(st.session_state['excel_results']) > 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for res in st.session_state['excel_results']:
                    file_content = res['html'] if isinstance(res['html'], bytes) else res['html'].encode('utf-8')
                    zf.writestr(res['filename'], file_content)
            
            st.download_button(
                label="📦 DOWNLOAD ALL REPORTS (.ZIP)",
                data=zip_buffer.getvalue(),
                file_name="BilingualDiff_All_Reports.zip",
                mime="application/zip",
                key="download_all_zip"
            )
            st.divider()

        for i, res in enumerate(st.session_state['excel_results']):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{res['original_name']}**")
            with col_b:
                with st.expander("Stats"):
                     st.write(f"Changed: {res['stats']['changed']} ({res['stats']['pct']:.1f}%)")
                     st.write(f"Corpus TER: {res['stats']['corpus_ter']:.1f}%")

            file_content = res['html'] if isinstance(res['html'], bytes) else res['html'].encode('utf-8')
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if res['filename'].endswith('.xlsx') else "text/html"
            st.download_button(
                label=f"⬇️ DOWNLOAD REPORT ({res['filename']})",
                data=file_content,
                file_name=res['filename'],
                mime=mime_type,
                key=f"dl_btn_{i}" 
            )
            st.markdown("---")

# 3. WOL REPORT MODE
elif mode == "WOL Report":
    if 'excel_results' in st.session_state: del st.session_state['excel_results']

    v1_file = st.file_uploader("Upload WOL HTML Report", type=["html", "htm"])
    if st.button("Compare & Generate Report"):
        if not v1_file:
            st.warning("Please upload the file.")
        else:
            with st.spinner("Processing..."):
                try:
                    v1_segs = load_segments(v1_file, True)
                    v2_segs = load_segments(v1_file, False)
                    file_data, stats, out_filename, mime_type = generate_report_based_on_toggle(
                        compare_segments(v1_segs, v2_segs), filter_map[filter_opt], export_format, mode, v1_file
                    )
                    st.success("Comparison Complete!")
                    st.download_button(label=f"⬇️ DOWNLOAD REPORT ({out_filename})", data=file_data, file_name=out_filename, mime=mime_type)
                except Exception as e:
                    st.error(f"Error: {e}")

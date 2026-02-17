import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import html
import re
from difflib import ndiff, SequenceMatcher
from collections import Counter
import io
import os

# ==========================================
# PART 1: CORE PARSING LOGIC
# ==========================================

def parse_tmx(file_content):
    segments = []
    tree = ET.parse(file_content)
    root = tree.getroot()
    ns = {'xml': 'http://www.w3.org/XML/1998/namespace'}
    
    # 1. Try to detect source language from the header
    header = root.find("header")
    src_lang = "en" # Default fallback
    if header is not None:
        src_lang = header.attrib.get("srclang", "en").lower()

    for tu in root.iter("tu"):
        source_text = ""
        target_text = ""
        
        tuvs = tu.findall("tuv")
        if len(tuvs) < 2: continue # Skip if not a pair

        # Strategy 1: Use TMX 'srclang' to identify source
        for tuv in tuvs:
            lang = tuv.attrib.get(f"{{{ns['xml']}}}lang", "").lower()
            seg = tuv.find("seg")
            text = seg.text if seg is not None else ""
            
            if not text: continue

            # If this TUV matches the header source language, it's Source.
            # Otherwise, it's Target.
            # We use 'in' because sometimes codes are 'en-US' vs 'en'
            if src_lang in lang:
                source_text = text
            else:
                target_text = text
        
        # Strategy 2: Fallback (First TUV is Source, Second is Target)
        if not source_text or not target_text:
             # Reset and force order
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
    elif filename.endswith(".mxliff") or filename.endswith(".sdlxliff"):
        return parse_xliff(uploaded_file)
    else:
        return []

# ==========================================
# PART 2: EXPORT & COMPARE LOGIC
# ==========================================

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
        lang_match = re.search(r'([a-z]{2}[-_][a-z]{2})', name1, re.IGNORECASE)
        lang_code = lang_match.group(1).lower() if lang_match else "unknown"
        part1 = name1[:15]
        part2 = name2[:15]
        return f"{lang_code}_{part1}_vs_{part2}.html"
    elif mode in ["Excel (3 Columns)", "WOL Report"] and v1_file:
        base_name = os.path.splitext(v1_file.name)[0]
        return f"{base_name}_Report.html"
    return "BilingualDiff_Report.html"

def generate_html_report(v1_segs, v2_segs, filter_option):
    filtered = []
    
    total_strings = len(v1_segs)
    changed_strings = 0
    removed_words_counter = Counter()
    added_words_counter = Counter()
    
    edit_distances = [] 
    total_len_v1 = 0
    total_len_v2 = 0
    
    for i, (seg1, seg2) in enumerate(zip(v1_segs, v2_segs), 1):
        source = seg1.get("source", "")
        v1 = seg1.get("target", "")
        v2 = seg2.get("target", "")
        
        total_len_v1 += len(v1)
        total_len_v2 += len(v2)

        status = "Same"
        score = 100 

        if v1 != v2:
            status = "Different"
            changed_strings += 1
            
            matcher = SequenceMatcher(None, v1, v2)
            score = round(matcher.ratio() * 100, 1)
            edit_distances.append(score)
            
            w1 = Counter(get_valid_words(v1))
            w2 = Counter(get_valid_words(v2))
            
            removed = (w1 - w2).elements()
            removed_words_counter.update(removed)
            
            added = (w2 - w1).elements()
            added_words_counter.update(added)
        else:
             edit_distances.append(100)

        if filter_option == "diff" and status == "Same": continue
        if filter_option == "same" and status == "Different": continue

        v1_hl, v2_hl = highlight_differences(v1, v2)
        filtered.append((i, source, v1_hl, v2_hl, status, score))

    change_pct = (changed_strings / total_strings * 100) if total_strings > 0 else 0
    top5_removed = removed_words_counter.most_common(5)
    top5_added = added_words_counter.most_common(5)
    expansion = ((total_len_v2 - total_len_v1) / total_len_v1 * 100) if total_len_v1 > 0 else 0
    
    changed_scores = [s for s in edit_distances if s < 100]
    avg_edit_score = sum(changed_scores) / len(changed_scores) if changed_scores else 100

    edit_categories = {"Minor Edit (>85%)": 0, "Medium Edit (50-85%)": 0, "Major Rewrite (<50%)": 0}
    for s in changed_scores:
        if s > 85: edit_categories["Minor Edit (>85%)"] += 1
        elif s >= 50: edit_categories["Medium Edit (50-85%)"] += 1
        else: edit_categories["Major Rewrite (<50%)"] += 1

    def list_to_html(lst):
        return ", ".join([f"{w} ({c})" for w, c in lst]) if lst else "None"

    html_out = f"""
    <html><head><meta charset="UTF-8"><title>BilingualDiff Report</title>
    <style>
        body {{ font-family: Segoe UI, sans-serif; padding: 20px; color: #333; }} 
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }} 
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }} 
        th {{ background-color: #f2f2f2; text-align: left; }} 
        .diff {{ background-color: #fff9db; }} 
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .stat-val {{ font-size: 1.2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; color: #666; }}
    </style>
    </head><body>
    <h2>Comparison Report</h2>
    
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-val">{total_strings}</div><div class="stat-label">Total Strings</div></div>
        <div class="stat-card"><div class="stat-val">{changed_strings} ({change_pct:.1f}%)</div><div class="stat-label">Changed Strings</div></div>
        <div class="stat-card"><div class="stat-val">{expansion:+.1f}%</div><div class="stat-label">Expansion Factor</div></div>
        <div class="stat-card"><div class="stat-val">{avg_edit_score:.1f}%</div><div class="stat-label">Avg Similarity (on edits)</div></div>
    </div>
    
    <p><b>Top Removed:</b> {list_to_html(top5_removed)}<br>
    <b>Top Added:</b> {list_to_html(top5_added)}</p>

    <table>
    <tr><th>ID</th><th>Source</th><th>Original Version</th><th>Updated Version</th><th>Match %</th></tr>
    """
    
    for seg_id, src, v1, v2, status, score in filtered:
        row_class = "diff" if status == "Different" else ""
        html_out += f'<tr class="{row_class}"><td>{seg_id}</td><td>{src}</td><td>{v1}</td><td>{v2}</td><td>{score}%</td></tr>'
    
    html_out += "</table></body></html>"
    
    stats = {
        "total": total_strings,
        "changed": changed_strings,
        "pct": change_pct,
        "expansion": expansion,
        "avg_score": avg_edit_score,
        "top_removed": top5_removed,
        "top_added": top5_added,
        "graph_data": edit_categories
    }
    
    return html_out, stats

# ==========================================
# PART 3: STREAMLIT UI
# ==========================================

st.set_page_config(page_title="BilingualDiff", layout="wide")

# Corrected CSS for Green Download Button
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

v1_file = None
v2_file = None

if mode == "Bilingual Files (TMX/XLIFF)":
    col1, col2 = st.columns(2)
    v1_file = col1.file_uploader("Upload Original Version", type=["tmx", "mxliff", "sdlxliff"])
    v2_file = col2.file_uploader("Upload Updated Version", type=["tmx", "mxliff", "sdlxliff"])

elif mode == "Excel (3 Columns)":
    v1_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    st.info("Excel must have Source in Col A, Original in Col B, Updated in Col C.")

elif mode == "WOL Report":
    v1_file = st.file_uploader("Upload WOL HTML Report", type=["html", "htm"])

if st.button("Compare & Generate Report"):
    if not v1_file:
        st.warning("Please upload the required files.")
    else:
        with st.spinner("Processing in browser..."):
            try:
                if mode == "Excel (3 Columns)":
                    v1_segs, v2_segs = parse_excel(v1_file)
                elif mode == "WOL Report":
                    v1_segs = load_segments(v1_file, True)
                    v2_segs = load_segments(v1_file, False)
                else: 
                    if not v2_file:
                        st.error("Updated Version file is missing!")
                        st.stop()
                    v1_segs = load_segments(v1_file, True)
                    v2_segs = load_segments(v2_file, False)
                
                report_html, stats = generate_html_report(v1_segs, v2_segs, filter_map[filter_opt])
                
                st.success("Comparison Complete!")
                
                out_filename = generate_output_filename(mode, v1_file, v2_file)
                
                # Big Green Download Button
                st.download_button(
                    label=f"⬇️ DOWNLOAD REPORT ({out_filename})",
                    data=report_html,
                    file_name=out_filename,
                    mime="text/html"
                )
                
                st.divider()
                st.subheader("📊 Translation Analytics")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Strings", stats["total"], help="Total count of segments found in the files.")
                m2.metric("Changed Strings", f"{stats['changed']} ({stats['pct']:.1f}%)", help="Number of segments where text differs between versions.")
                m3.metric("Expansion Factor", f"{stats['expansion']:+.1f}%", help="Length difference (Positive = Updated is longer).")
                m4.metric("Avg Edit Similarity", f"{stats['avg_score']:.1f}%", help="Levenshtein score (100% = Identical, 0% = Total Rewrite).")
                
                st.divider()
                
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown("#### Edit Intensity Distribution")
                    st.caption("Categorizes changes by how much the text was altered (Minor vs Major).")
                    
                    categories = list(stats["graph_data"].keys())
                    counts = list(stats["graph_data"].values())
                    
                    if sum(counts) > 0:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        bars = ax.bar(categories, counts, color=['#4caf50', '#ff9800', '#f44336'])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.set_ylabel("Count")
                        plt.xticks(rotation=15, ha='right')
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3), 
                                        textcoords="offset points",
                                        ha='center', va='bottom')
                        
                        st.pyplot(fig)
                    else:
                        st.info("No changes detected to graph.")

                with c2:
                    st.markdown("#### Most Removed Words")
                    st.caption("Words frequently found in Original but gone in Updated.")
                    if stats['top_removed']:
                        for w, c in stats['top_removed']:
                            st.markdown(f"- **{w}**: {c}x")
                    else: st.markdown("_None_")

                    st.markdown("#### Most Added Words")
                    st.caption("Words frequently found in Updated but not in Original.")
                    if stats['top_added']:
                        for w, c in stats['top_added']:
                            st.markdown(f"- **{w}**: {c}x")
                    else: st.markdown("_None_")
                
                st.divider()
                
                st.subheader("Preview (First 5 Differences)")
                count = 0
                preview_shown = False
                for s1, s2 in zip(v1_segs, v2_segs):
                    if s1['target'] != s2['target']:
                        st.text(f"Source: {s1['source'][:50]}...")
                        st.markdown(f"**Original:** {s1['target']}")
                        st.markdown(f"**Updated:** {s2['target']}")
                        st.divider()
                        count += 1
                        preview_shown = True
                        if count >= 5: break
                
                if not preview_shown:
                    st.info("No differences found to preview.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

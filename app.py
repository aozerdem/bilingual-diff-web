import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import html
import re
from difflib import ndiff
from collections import Counter
import io

# ==========================================
# PART 1: CORE PARSING LOGIC (Refactored)
# ==========================================

def parse_tmx(file_content):
    segments = []
    tree = ET.parse(file_content)
    root = tree.getroot()
    ns = {'xml': 'http://www.w3.org/XML/1998/namespace'}
    
    for tu in root.iter("tu"):
        seg_pair = {"source": "", "target": ""}
        for tuv in tu.findall("tuv"):
            # Handle xml:lang attribute correctly
            lang = tuv.attrib.get(f"{{{ns['xml']}}}lang", "").lower()
            seg = tuv.find("seg")
            if seg is not None:
                text = seg.text or ""
                # Simple heuristic: assuming en-us is source (adjust as needed)
                if "en" in lang: 
                    seg_pair["source"] = text
                else:
                    seg_pair["target"] = text
        
        if seg_pair["source"] and seg_pair["target"]:
            segments.append(seg_pair)
    return segments

def parse_excel(file_content):
    # Engine 'openpyxl' is required for xlsx
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
        # Detect namespace
        ns_match = re.match(r'\{(.*)\}', root.tag)
        ns = {'ns': ns_match.group(1)} if ns_match else {}
        
        # Handle both mxliff and sdlxliff logic
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

def parse_wol_html(file_content, is_version1):
    segments = []
    # Use 'lxml' if available, but 'html.parser' is standard in stlite
    soup = BeautifulSoup(file_content, 'html.parser')
    tables = soup.find_all("table")
    
    target_table = None
    headers = []
    
    # 1. SMART TABLE DETECTION
    # Loop through all tables to find the one containing the correct headers
    for table in tables:
        # Get the first row (header row)
        rows = table.find_all("tr")
        if not rows:
            continue
            
        # Extract text from all cells (th or td) in the first row
        # We use separator=" " to ensure "PE" and "Translation" don't get stuck together
        cells = rows[0].find_all(['th', 'td'])
        current_headers = [c.get_text(" ", strip=True).lower() for c in cells]
        
        # Check if this looks like the data table (must have "source" and "re translation")
        if "source" in current_headers and "re translation" in current_headers:
            target_table = table
            headers = current_headers
            break
            
    if target_table is None:
        st.error("Could not find the Data Table in the uploaded WOL Report. Please check the file format.")
        return []

    # 2. FLEXIBLE COLUMN INDEXING
    try:
        source_idx = headers.index("source")
        re_idx = headers.index("re translation")
        
        # Look for the Version 1 column (PE, HT, or just Translation)
        pe_idx = -1
        possible_v1_headers = ["pe translation", "ht translation", "mt translation", "translation"]
        
        for h in possible_v1_headers:
            if h in headers:
                pe_idx = headers.index(h)
                break
        
        if pe_idx == -1:
             st.error("Could not find 'PE Translation' or 'HT Translation' column.")
             return []

    except ValueError:
        st.error("Error identifying column indexes.")
        return []

    # 3. EXTRACT DATA
    # Skip the header row ([1:])
    for row in target_table.find_all("tr")[1:]:
        cells = row.find_all("td")
        # Ensure the row has enough cells
        if len(cells) > max(source_idx, pe_idx, re_idx):
            source = html.unescape(cells[source_idx].get_text(" ", strip=True))
            v1 = html.unescape(cells[pe_idx].get_text(" ", strip=True))
            v2 = html.unescape(cells[re_idx].get_text(" ", strip=True))
            
            segments.append({"source": source, "target": v1 if is_version1 else v2})
            
    return segments

def load_segments(uploaded_file, is_version1=True):
    # Streamlit uploads are BytesIO objects. We reset cursor to 0 just in case.
    uploaded_file.seek(0)
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".tmx"):
        return parse_tmx(uploaded_file)
    elif filename.endswith(".xlsx"):
        return parse_excel(uploaded_file)[0] if is_version1 else parse_excel(uploaded_file)[1]
    elif filename.endswith(".htm") or filename.endswith(".html"):
        return parse_wol_html(uploaded_file, is_version1)
    elif filename.endswith(".mxliff") or filename.endswith(".sdlxliff"):
        return parse_xliff(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return []

# ==========================================
# PART 2: EXPORT & COMPARE LOGIC (Refactored)
# ==========================================

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

def generate_html_report(v1_segs, v2_segs, filter_option):
    filtered = []
    major_changes = 0
    total_source_words = 0
    
    for i, (seg1, seg2) in enumerate(zip(v1_segs, v2_segs), 1):
        source = seg1.get("source", "")
        v1 = seg1.get("target", "")
        v2 = seg2.get("target", "")
        
        total_source_words += len(source.split())

        if v1 == v2:
            status = "Same"
            if filter_option == "diff": continue
        else:
            status = "Different"
            change_ratio = sum(1 for a, b in zip(v1, v2) if a != b) / max(len(v1), 1)
            if change_ratio >= 0.5: major_changes += 1
        
        if filter_option == "same" and status == "Different": continue

        v1_hl, v2_hl = highlight_differences(v1, v2)
        filtered.append((i, source, v1_hl, v2_hl, status))

    # Create HTML String
    html_out = f"""
    <html><head><meta charset="UTF-8"><title>BilingualDiff Report</title>
    <style>body {{ font-family: Segoe UI, sans-serif; padding: 20px; }} table {{ width: 100%; border-collapse: collapse; }} th, td {{ border: 1px solid #ddd; padding: 8px; }} th {{ background-color: #f2f2f2; }} .diff {{ background-color: #fff3cd; }} </style>
    </head><body>
    <h2>Comparison Report</h2>
    <p><b>Total Words:</b> {total_source_words} | <b>Major Changes:</b> {major_changes}</p>
    <table>
    <tr><th>ID</th><th>Source</th><th>Version 1</th><th>Version 2</th><th>Status</th></tr>
    """
    
    for seg_id, src, v1, v2, status in filtered:
        row_class = "diff" if status == "Different" else ""
        html_out += f'<tr class="{row_class}"><td>{seg_id}</td><td>{src}</td><td>{v1}</td><td>{v2}</td><td>{status}</td></tr>'
    
    html_out += "</table></body></html>"
    return html_out

# ==========================================
# PART 3: STREAMLIT UI
# ==========================================

st.set_page_config(page_title="BilingualDiff", layout="wide")

st.title("🌍 BilingualDiff (Secure Web Edition)")
st.markdown("Compare translation versions securely. **Files never leave your browser.**")

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Select Mode", ["Bilingual Files (TMX/XLIFF)", "Excel (3 Columns)", "WOL Report"])
    filter_opt = st.radio("Export Filter", ["All Segments", "Only DIFFERENT", "Only SAME"], index=0)
    filter_map = {"All Segments": "all", "Only DIFFERENT": "diff", "Only SAME": "same"}

v1_file = None
v2_file = None

# UI Logic
if mode == "Bilingual Files (TMX/XLIFF)":
    col1, col2 = st.columns(2)
    v1_file = col1.file_uploader("Upload Version 1", type=["tmx", "mxliff", "sdlxliff"])
    v2_file = col2.file_uploader("Upload Version 2", type=["tmx", "mxliff", "sdlxliff"])

elif mode == "Excel (3 Columns)":
    v1_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    st.info("Excel must have Source in Col A, Version 1 in Col B, Version 2 in Col C.")

elif mode == "WOL Report":
    v1_file = st.file_uploader("Upload WOL HTML Report", type=["html", "htm"])

# Processing Logic
if st.button("Compare & Generate Report"):
    if not v1_file:
        st.warning("Please upload the required files.")
    else:
        with st.spinner("Processing in browser..."):
            try:
                if mode == "Excel (3 Columns)":
                    v1_segs, v2_segs = parse_excel(v1_file)
                elif mode == "WOL Report":
                    # We need to read the stream twice, so we create a copy in memory
                    # But parse_wol reads the whole soup, so we can just pass the stream
                    # Actually, load_segments resets the stream seek(0) so we are good.
                    v1_segs = load_segments(v1_file, True)
                    v2_segs = load_segments(v1_file, False)
                else: # TMX/XLIFF
                    if not v2_file:
                        st.error("Version 2 file is missing!")
                        st.stop()
                    v1_segs = load_segments(v1_file, True)
                    v2_segs = load_segments(v2_file, False)
                
                # Generate Report
                report_html = generate_html_report(v1_segs, v2_segs, filter_map[filter_opt])
                
                st.success("Comparison Complete!")
                
                # Preview
                st.subheader("Preview (First 5 Differences)")
                # Simple preview logic
                count = 0
                for s1, s2 in zip(v1_segs, v2_segs):
                    if s1['target'] != s2['target']:
                        st.text(f"Source: {s1['source'][:50]}...")
                        st.markdown(f"**V1:** {s1['target']}")
                        st.markdown(f"**V2:** {s2['target']}")
                        st.divider()
                        count += 1
                        if count >= 5: break
                
                # Download Button
                st.download_button(
                    label="Download Full HTML Report",
                    data=report_html,
                    file_name="BilingualDiff_Report.html",
                    mime="text/html"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
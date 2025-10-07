# ==============================================================
# 🚀 Company–Domain–Email Matching (Railway & GitHub Ready)
# ==============================================================

import os, asyncio, re, unicodedata, tldextract, aiohttp, nest_asyncio, requests_cache
import pandas as pd
import numpy as np
import gradio as gr
from rapidfuzz import fuzz

# Enable nested event loops (for hosted environments)
nest_asyncio.apply()
requests_cache.install_cache("web_cache", expire_after=86400)

THRESHOLD_STRONG, THRESHOLD_MID = 85, 70


# ==============================================================
# 🧰 Utility Functions
# ==============================================================

def normalize_text(s):
    """Basic text normalization"""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s


def get_headers(file):
    """Return all column headers regardless of file format."""
    ext = os.path.splitext(file.name)[1].lower()

    if ext in [".xls", ".xlsx"]:
        xls = pd.ExcelFile(file.name)
        first_sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=first_sheet, nrows=1)
    elif ext == ".csv":
        try:
            df = pd.read_csv(file.name, nrows=1)
        except Exception:
            df = pd.read_csv(file.name, nrows=1, sep=";")
    elif ext in [".tsv", ".txt"]:
        df = pd.read_csv(file.name, sep="\t", nrows=1)
    elif ext == ".parquet":
        df = pd.read_parquet(file.name)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return df.columns.tolist()


def read_any(file):
    """Read full dataset and any additional Excel sheets."""
    ext = os.path.splitext(file.name)[1].lower()

    if ext in [".xls", ".xlsx"]:
        xls = pd.ExcelFile(file.name)
        first_sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=first_sheet)
        other_sheets = {s: pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names[1:]}
        return df, other_sheets, first_sheet

    elif ext == ".csv":
        try:
            df = pd.read_csv(file.name)
        except Exception:
            df = pd.read_csv(file.name, sep=";")
        return df, {}, "Sheet1"

    elif ext in [".tsv", ".txt"]:
        df = pd.read_csv(file.name, sep="\t")
        return df, {}, "Sheet1"

    elif ext == ".parquet":
        df = pd.read_parquet(file.name)
        return df, {}, "Sheet1"

    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ==============================================================
# 🧠 Matching Logic (Simplified Placeholder)
# ==============================================================

# ⚠️ Replace this section later with your full logic (run_match, add_email_name_flags, add_confidence)

async def run_match(df, company_col, domain_col):
    """Simulated async match logic (replace later with your real one)."""
    if company_col not in df.columns or domain_col not in df.columns:
        raise ValueError("Selected columns not found in file.")

    out = df.copy()
    out["Match_Confidence"] = np.random.randint(70, 100, size=len(out))
    out["Validated_Email"] = np.where(
        out[domain_col].astype(str).str.contains(r"\.", na=False), "✅", "❌"
    )
    return out


def add_email_name_flags(df):
    df["Email_Name_Flag"] = np.where(
        df["Validated_Email"] == "✅", "Match OK", "Missing/Invalid"
    )
    return df


def add_confidence(df):
    df["Confidence_Level"] = np.where(
        df["Match_Confidence"] >= 85, "High", "Medium"
    )
    return df


# ==============================================================
# ⚙️ Main Processing Function
# ==============================================================

def process_with_selection(file, company_col, domain_col):
    df, other_sheets, first_sheet = read_any(file)

    # Run async logic
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_match(df, company_col, domain_col))
    result = add_email_name_flags(result)
    result = add_confidence(result)

    base = os.path.splitext(os.path.basename(file.name))[0]
    out_path = f"{base}_matched.xlsx"

    # Save as Excel, preserving extra sheets if any
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name=first_sheet, index=False)
        for name, sheet_df in other_sheets.items():
            sheet_df.to_excel(writer, sheet_name=name, index=False)

    return (
        f"✅ Matching complete!\nCompany: {company_col}\nDomain: {domain_col}\nSaved: {out_path}",
        out_path,
    )


# ==============================================================
# 🎨 Gradio Front-End
# ==============================================================

with gr.Blocks(title="Company–Domain–Email Matching") as demo:
    gr.Markdown(
        """
        ## 🧠 Company–Domain–Email Matching Tool  
        Upload any data file (Excel, CSV, TSV, or Parquet), choose which columns contain **Company** and **Domain**,  
        and download an enriched `.xlsx` with validation and confidence levels.
        """
    )

    file_input = gr.File(label="📁 Upload File")
    load_cols_btn = gr.Button("🔍 Load Columns")

    company_dropdown = gr.Dropdown(label="Select Company Column", choices=[], interactive=True)
    domain_dropdown = gr.Dropdown(label="Select Domain Column", choices=[], interactive=True)

    run_btn = gr.Button("🚀 Run Matching")
    log = gr.Textbox(label="🪄 Process Log", interactive=False)
    download = gr.File(label="⬇️ Download Updated Workbook (.xlsx)")

    # Load columns dynamically
    def load_columns(file):
        cols = get_headers(file)
        auto_company = next((c for c in cols if "company" in c.lower()), cols[0] if cols else None)
        auto_domain = next(
            (c for c in cols if any(k in c.lower() for k in ["domain", "website", "email", "url"])),
            cols[-1] if cols else None
        )
        return (
            gr.update(choices=cols, value=auto_company),
            gr.update(choices=cols, value=auto_domain)
        )

    load_cols_btn.click(fn=load_columns, inputs=file_input, outputs=[company_dropdown, domain_dropdown])
    run_btn.click(fn=process_with_selection, inputs=[file_input, company_dropdown, domain_dropdown], outputs=[log, download])

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

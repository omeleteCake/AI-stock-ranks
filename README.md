# AI-Stock-Ranks

This repository contains a framework for ranking publicly listed stocks using a
listwise, relative-comparison protocol and a large language model (LLM). The
design follows a batch-and-average methodology: tickers are grouped into small
batches, ranked relative to each other by an LLM, and aggregated over multiple
trials to produce stable, context-aware ratings.

The code in `src/` implements the core logic for creating random batches,
calling an LLM to rank each batch, averaging the results, and producing a final
ordered list of tickers. The `docs/` folder defines the product requirements,
data contracts, prompts, and evaluation procedures.

## PDF Input Mode

You can run the orchestrator from PDF documents (financial statements and
investor presentations) instead of a prebuilt CSV.

Recommended layout:

```text
pdf_inputs/
  AAPL/
    financial_statement.pdf
    investor_presentation.pdf
  MSFT/
    10k.pdf
    investor_day.pdf
```

Run:

```bash
python -m src.orchestrator pdf_inputs --input-mode pdf
```

The orchestrator will:
1. Parse PDFs and extract ticker metrics into the ranker schema.
2. Save the derived input table in `outputs/runs/<timestamp>/input_from_pdf.csv`.
3. Run the normal batch-and-average ranking pipeline.

## SEC Fetch Script

Use the SEC fetch script to build your document input folder automatically from
recent filings (10-Q/10-K/20-F/6-K/8-K).

```bash
python -m src.fetch_latest_sec_pdf --tickers AAPL,MSFT,NVDA --out-dir pdf_inputs
```

Or with a CSV ticker universe:

```bash
python -m src.fetch_latest_sec_pdf --tickers-csv universe.csv --out-dir pdf_inputs
```

Then run ranking:

```bash
python -m src.orchestrator pdf_inputs --input-mode pdf
```

If a filing has no PDF attachment, the fetcher falls back to the filing's
primary HTML/TXT document so the LLM can still read the latest filing content.

Set a SEC-compliant user agent (PowerShell):

```bash
$env:SEC_USER_AGENT = "AIStockRanks/1.0 (contact: your-email@example.com)"
```

## Assessment Website UI

A browser dashboard is available in `web/` to inspect ranking outputs.

Start a local server from repo root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/web/
```

In the UI, upload either:
1. `outputs/runs/<timestamp>/aggregate.json`
2. `outputs/runs/<timestamp>/aggregate.csv`

The dashboard provides filters, stability scatter plot, top conviction list,
caution list, and full ranked table.

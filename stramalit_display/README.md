# Strategic Analytics Display

This folder contains the Streamlit interface for the HackNova analytics pipeline.

## Purpose

This app is **view-only**.

It is designed to visualize and explain analysis results. It does **not**:
- place trades
- execute brokerage actions
- save live portfolio decisions

## Pages Included

1. **Risk Return Analysis (Task 2)**
2. **Technical Signals (Task 3)**
3. **Portfolio Optimization (Task 4)**
4. **Chaos Stress Test (Task 5)**

## Setup

```bash
cd stramalit_display
pip install -r requirements.txt
streamlit run dashboard.py
```

## Data Source

The dashboard reads from:

- `data/cleaned_data/`

If needed, refresh from the project root cleaned data:

```bash
cd ..
mkdir -p stramalit_display/data
cp -r cleaned_data stramalit_display/data/
```

## Folder Notes

- `dashboard.py`: main Streamlit app
- `requirements.txt`: Streamlit app dependencies
- `data/cleaned_data/`: local app input data
- `src/`: optional module/code copies used during development

## Recommendation

For complete project documentation and CLI run order, use the root-level README at `../README.md`.
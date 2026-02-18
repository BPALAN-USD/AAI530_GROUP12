## Smart Parking Analytics (Notebook)

This project centers around the notebook `SmartParking.ipynb`, which analyzes smart-parking session data from monthly CSV exports in `Data/`.

### What the notebook shows

- **Load & explore parking sessions** from local CSVs (no Snowflake dependency).
- **Exploratory Data Analysis (EDA)**:
  - dataset shape, schema, basic stats
  - missing value analysis and distributions
- **Data cleaning**:
  - remove rows missing `OUT_TIME`
  - drop ingestion/offset columns that aren’t needed for analytics
- **Parking occupancy analysis**:
  - derive occupancy over time using entry (+1) and exit (-1) events
  - visualize occupancy and arrivals-by-hour patterns
- **ML models**:
  - **Model 1**: predict parking duration (Linear Regression → Random Forest → enhanced with facility/district encodings)
  - **Model 2**: predict peak occupancy (Linear Regression → Random Forest with weekend/rush-hour features)
  - **Model 3**: predict parking fee/cost (Linear Regression → Random Forest with non-linear duration feature)
- **Deep Learning (optional/heavier)**:
  - LSTM time-series forecasting of hourly occupancy

### Data required

Place monthly CSVs in:

- `Data/parking_sessions_*.csv`

Expected columns (as provided in the CSVs):

- `SESSION_ID`, `LICENSE_PLATE`, `FACILITY_ID`, `FACILITY_NAME`, `DISTRICT`
- `IN_TIME`, `OUT_TIME`
- `ACTUAL_DURATION_HOURS`, `RATE_PER_HOUR`, `COST`, `STATUS`
- `LICENSE_PLATE_STATE`, `INGESTED_AT`, `ROW_TIMESTAMP`, `OFFSET_ID`

The notebook concatenates all matching CSVs into a single dataframe named `dataframe_1`.

---
## Running / Deployment Instructions

You have two common ways to run the notebook:

- **VS Code / Cursor notebook UI** (recommended)
- **Jupyter Lab** from the terminal

Either way, use the project virtual environment as the kernel.

### 1) Create the Python environment

From the `iot_project/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2) Register the kernel for Jupyter

Still with the venv activated:

```bash
python -m ipykernel install --user --name smartparking --display-name "Python (smartparking)"
```

Verify:

```bash
python -m jupyter kernelspec list
```

You should see an entry like `smartparking`.

### 3A) Run in VS Code / Cursor

- Open `SmartParking.ipynb`
- Use the **kernel picker** and select **Python (smartparking)**
- Run cells top-to-bottom

### 3B) Run with Jupyter Lab

With the venv activated:

```bash
python -m pip install jupyterlab
jupyter lab
```

Then open `SmartParking.ipynb` and select **Python (smartparking)** as the kernel.

---
## Troubleshooting

### Mixed timezones detected (pandas `to_datetime` ValueError)

The CSV timestamps include offsets (e.g. `-0500` and `-0400`). The notebook normalizes by parsing with `utc=True` and converting to a single timezone before dropping tz info.

If you reintroduce parsing elsewhere, use the same pattern:

- `pd.to_datetime(..., utc=True)` then convert to a consistent timezone (e.g. `America/New_York`)

### pip SSL certificate verification errors on macOS (Homebrew Python)

If installs fail with an `SSLCertVerificationError`, your local CA trust may be out of sync. A workaround is to install using trusted hosts:

```bash
pip install -r requirements.txt \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host pypi.python.org
```

Long-term fix is to repair certificate trust for your Python/OpenSSL install (Homebrew/OpenSSL paths).

### TensorFlow install/runtime notes (Apple Silicon)

- TensorFlow is large and may take time to install.
- For best performance on Apple Silicon, ensure you’re using arm64 wheels (this project does).


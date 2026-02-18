from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


NY_TZ = "America/New_York"


@dataclass(frozen=True)
class Models:
    duration_model: RandomForestRegressor
    cost_model: RandomForestRegressor
    occupancy_model: RandomForestRegressor
    le_facility: LabelEncoder
    le_district: LabelEncoder
    duration_q95: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_raw_sessions(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted(data_dir.glob("parking_sessions_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found at: {data_dir.resolve()}/parking_sessions_*.csv")

    dfs = [
        pd.read_csv(
            f,
            na_values=["\\N"],
            keep_default_na=True,
        )
        for f in csv_files
    ]
    df = pd.concat(dfs, ignore_index=True)
    return df


def _parse_ny_naive(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    ts = ts.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    return ts


@st.cache_data(show_spinner=False)
def prepare_sessions(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "OUT_TIME" in df.columns:
        df = df.dropna(subset=["OUT_TIME"])

    for col in ["ROW_TIMESTAMP", "OFFSET_ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["IN_TIME"] = _parse_ny_naive(df["IN_TIME"])
    df["OUT_TIME"] = _parse_ny_naive(df["OUT_TIME"])
    df = df.dropna(subset=["IN_TIME", "OUT_TIME"])

    df["duration_minutes"] = (df["OUT_TIME"] - df["IN_TIME"]).dt.total_seconds() / 60.0
    df = df[df["duration_minutes"] >= 0].copy()

    df["arrival_hour"] = df["IN_TIME"].dt.hour
    df["day_of_week"] = df["IN_TIME"].dt.dayofweek
    df["month"] = df["IN_TIME"].dt.month
    df["date"] = df["IN_TIME"].dt.date

    return df


@st.cache_data(show_spinner=False)
def compute_events(df: pd.DataFrame) -> pd.DataFrame:
    events = pd.concat(
        [
            df[["IN_TIME"]].rename(columns={"IN_TIME": "time"}).assign(change=1),
            df[["OUT_TIME"]].rename(columns={"OUT_TIME": "time"}).assign(change=-1),
        ],
        ignore_index=True,
    ).sort_values("time")

    events["occupancy"] = events["change"].cumsum()
    return events


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> Models:
    df_nonneg = df[df["duration_minutes"] >= 0].copy()
    q95 = float(df_nonneg["duration_minutes"].quantile(0.95))
    df_clean = df_nonneg[df_nonneg["duration_minutes"] <= q95].copy()

    le_facility = LabelEncoder()
    le_district = LabelEncoder()

    df_clean["facility_encoded"] = le_facility.fit_transform(df_clean["FACILITY_ID"].astype(str))
    df_clean["district_encoded"] = le_district.fit_transform(df_clean["DISTRICT"].astype(str))

    X_duration = df_clean[
        ["arrival_hour", "day_of_week", "month", "facility_encoded", "district_encoded"]
    ]
    y_duration = df_clean["duration_minutes"]

    duration_model = RandomForestRegressor(
        n_estimators=250, max_depth=18, random_state=42, n_jobs=-1
    )
    duration_model.fit(X_duration, y_duration)

    df_fee = df_clean.dropna(subset=["COST"]).copy()
    df_fee["duration_squared"] = df_fee["duration_minutes"] ** 2
    X_fee = df_fee[["duration_minutes", "duration_squared", "arrival_hour", "day_of_week"]]
    y_fee = df_fee["COST"]

    cost_model = RandomForestRegressor(n_estimators=250, max_depth=14, random_state=42, n_jobs=-1)
    cost_model.fit(X_fee, y_fee)

    events = compute_events(df_nonneg)
    hourly = events.copy()
    hourly["hour"] = hourly["time"].dt.hour
    hourly["day_of_week"] = hourly["time"].dt.dayofweek
    hourly["month"] = hourly["time"].dt.month
    occ_agg = (
        hourly.groupby(["hour", "day_of_week", "month"], as_index=False)["occupancy"].max()
    )
    occ_agg["is_weekend"] = occ_agg["day_of_week"].isin([5, 6]).astype(int)
    occ_agg["is_rush_hour"] = occ_agg["hour"].isin([8, 9, 17, 18]).astype(int)

    X_occ = occ_agg[["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"]]
    y_occ = occ_agg["occupancy"]

    occupancy_model = RandomForestRegressor(
        n_estimators=300, max_depth=16, random_state=42, n_jobs=-1
    )
    occupancy_model.fit(X_occ, y_occ)

    return Models(
        duration_model=duration_model,
        cost_model=cost_model,
        occupancy_model=occupancy_model,
        le_facility=le_facility,
        le_district=le_district,
        duration_q95=q95,
    )


def _encode_or_none(le: LabelEncoder, value: str) -> Optional[int]:
    classes = set(le.classes_.tolist())
    if value not in classes:
        return None
    return int(le.transform([value])[0])


def predict_duration_minutes(models: Models, when: datetime, facility_id: str, district: str) -> float:
    facility_enc = _encode_or_none(models.le_facility, facility_id)
    district_enc = _encode_or_none(models.le_district, district)
    if facility_enc is None or district_enc is None:
        return float("nan")

    X = pd.DataFrame(
        [
            {
                "arrival_hour": when.hour,
                "day_of_week": when.weekday(),
                "month": when.month,
                "facility_encoded": facility_enc,
                "district_encoded": district_enc,
            }
        ]
    )
    return float(models.duration_model.predict(X)[0])


def predict_cost(models: Models, duration_minutes: float, when: datetime) -> float:
    X = pd.DataFrame(
        [
            {
                "duration_minutes": duration_minutes,
                "duration_squared": duration_minutes**2,
                "arrival_hour": when.hour,
                "day_of_week": when.weekday(),
            }
        ]
    )
    return float(models.cost_model.predict(X)[0])


def predict_occupancy(models: Models, when: datetime) -> float:
    hour = when.hour
    dow = when.weekday()
    month = when.month
    X = pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": dow,
                "month": month,
                "is_weekend": int(dow in (5, 6)),
                "is_rush_hour": int(hour in (8, 9, 17, 18)),
            }
        ]
    )
    return float(models.occupancy_model.predict(X)[0])


@st.cache_resource(show_spinner=False)
def _train_lstm_forecaster(
    occupancy_ts: pd.Series, seq_length: int = 24, epochs: int = 8
):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "TensorFlow is not installed in the Python environment running Streamlit. "
            "Activate the project venv and install dependencies, then re-run:\n\n"
            "  cd AAI530_GROUP12\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  streamlit run streamlit_app.py\n"
        ) from e

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(occupancy_ts.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i : i + seq_length])
        y.append(scaled[i + seq_length])
    X = np.array(X)
    y = np.array(y)

    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_length, 1)),
            tf.keras.layers.LSTM(48, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(48),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=0,
    )
    return model, scaler


def lstm_forecast_next_hours(events: pd.DataFrame, hours: int = 24) -> pd.Series:
    occupancy_ts = (
        events.set_index("time").resample("h")["occupancy"].mean().ffill()
    )
    occupancy_ts = occupancy_ts.dropna()
    if len(occupancy_ts) < 200:
        return pd.Series(dtype=float)

    seq_length = 24
    model, scaler = _train_lstm_forecaster(occupancy_ts, seq_length=seq_length, epochs=8)

    last = scaler.transform(occupancy_ts.values.reshape(-1, 1))[-seq_length:].reshape(
        1, seq_length, 1
    )
    preds = []
    cur = last.copy()
    for _ in range(hours):
        pred = model.predict(cur, verbose=0)[0][0]
        preds.append(pred)
        cur = np.concatenate([cur[:, 1:, :], np.array(pred).reshape(1, 1, 1)], axis=1)

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).reshape(-1)
    start = occupancy_ts.index.max() + pd.Timedelta(hours=1)
    idx = pd.date_range(start=start, periods=hours, freq="h")
    return pd.Series(preds_inv, index=idx, name="forecast_occupancy")


def rf_forecast_next_hours(models: Models, start_at: datetime, hours: int = 24) -> pd.Series:
    future = [start_at + timedelta(hours=i) for i in range(1, hours + 1)]
    rows = []
    for t in future:
        hour = t.hour
        dow = t.weekday()
        rows.append(
            {
                "hour": hour,
                "day_of_week": dow,
                "month": t.month,
                "is_weekend": int(dow in (5, 6)),
                "is_rush_hour": int(hour in (8, 9, 17, 18)),
            }
        )
    X = pd.DataFrame(rows)
    y = models.occupancy_model.predict(X)
    return pd.Series(y, index=pd.to_datetime(future), name="forecast_occupancy")


def main() -> None:
    st.set_page_config(page_title="Smart Parking — Executive Dashboard", layout="wide")

    st.title("Smart Parking — Executive Dashboard")
    st.caption("KPIs, predictions, and occupancy forecasting from parking session data.")

    data_dir = _repo_root() / "Data"

    with st.sidebar:
        st.header("Data")
        st.write(f"Data directory: `{data_dir}`")

    df_raw = load_raw_sessions(data_dir)
    df = prepare_sessions(df_raw)

    with st.sidebar:
        st.header("Filters")
        districts = sorted(df["DISTRICT"].dropna().astype(str).unique().tolist())
        facilities = (
            df[["FACILITY_ID", "FACILITY_NAME"]]
            .dropna()
            .assign(FACILITY_ID=lambda x: x["FACILITY_ID"].astype(str))
            .drop_duplicates()
            .sort_values(["FACILITY_NAME", "FACILITY_ID"])
        )

        default_districts = districts
        selected_districts = st.multiselect(
            "District", options=districts, default=default_districts
        )

        facility_labels = [
            f"{row.FACILITY_NAME} (ID {row.FACILITY_ID})" for row in facilities.itertuples()
        ]
        selected_facilities = st.multiselect(
            "Facility", options=facility_labels, default=facility_labels
        )

        min_d = df["IN_TIME"].min().date()
        max_d = df["IN_TIME"].max().date()
        d1, d2 = st.date_input("Date range (IN_TIME)", value=(min_d, max_d))
        if isinstance(d1, date) and isinstance(d2, date) and d2 < d1:
            d1, d2 = d2, d1

    facility_id_set = set()
    if selected_facilities:
        for label in selected_facilities:
            if "(ID " in label and label.endswith(")"):
                facility_id_set.add(label.split("(ID ", 1)[1][:-1])

    df_f = df.copy()
    if selected_districts:
        df_f = df_f[df_f["DISTRICT"].astype(str).isin(selected_districts)]
    if facility_id_set:
        df_f = df_f[df_f["FACILITY_ID"].astype(str).isin(facility_id_set)]
    if isinstance(d1, date) and isinstance(d2, date):
        df_f = df_f[(df_f["IN_TIME"].dt.date >= d1) & (df_f["IN_TIME"].dt.date <= d2)]

    events_f = compute_events(df_f)
    peak_occ = int(events_f["occupancy"].max()) if len(events_f) else 0

    total_sessions = int(len(df_f))
    avg_duration = float(df_f["duration_minutes"].mean()) if total_sessions else float("nan")
    total_revenue = float(df_f["COST"].fillna(0).sum()) if total_sessions else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sessions", f"{total_sessions:,}")
    c2.metric("Avg duration (min)", f"{avg_duration:,.1f}" if np.isfinite(avg_duration) else "—")
    c3.metric("Total revenue ($)", f"{total_revenue:,.0f}")
    c4.metric("Peak occupancy", f"{peak_occ:,} cars")

    tab_overview, tab_predictions, tab_forecast, tab_data = st.tabs(
        ["Overview", "Predictions", "Forecast", "Data"]
    )

    with tab_overview:
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Occupancy over time")
            if len(events_f):
                occ_ts = (
                    events_f.set_index("time").resample("h")["occupancy"].mean().ffill()
                )
                st.line_chart(occ_ts, height=280)
            else:
                st.info("No events available for the selected filters.")

            st.subheader("Revenue by day")
            if total_sessions:
                revenue_daily = (
                    df_f.assign(day=df_f["IN_TIME"].dt.floor("D"))
                    .groupby("day")["COST"]
                    .sum()
                )
                st.line_chart(revenue_daily, height=240)

        with right:
            st.subheader("Arrivals by hour")
            if total_sessions:
                arrivals = df_f.groupby("arrival_hour").size().reindex(range(24), fill_value=0)
                st.bar_chart(arrivals, height=300)

            st.subheader("Top facilities (sessions)")
            if total_sessions:
                top_fac = (
                    df_f.groupby("FACILITY_NAME").size().sort_values(ascending=False).head(10)
                )
                st.dataframe(top_fac.rename("sessions"), use_container_width=True)

    with tab_predictions:
        st.subheader("Scenario-based predictions")

        models = train_models(df)

        scenario_cols = st.columns([1, 1, 1, 1])
        sc_date = scenario_cols[0].date_input("Arrival date", value=date.today())
        sc_time = scenario_cols[1].time_input("Arrival time", value=time(9, 0))

        district_opt = sorted(df["DISTRICT"].dropna().astype(str).unique().tolist())
        sc_district = scenario_cols[2].selectbox(
            "District", options=district_opt, index=0 if district_opt else None
        )

        facility_opts = (
            df[["FACILITY_ID", "FACILITY_NAME"]]
            .dropna()
            .assign(FACILITY_ID=lambda x: x["FACILITY_ID"].astype(str))
            .drop_duplicates()
            .sort_values(["FACILITY_NAME", "FACILITY_ID"])
        )
        facility_labels2 = [
            f"{row.FACILITY_NAME} (ID {row.FACILITY_ID})" for row in facility_opts.itertuples()
        ]
        sc_fac_label = scenario_cols[3].selectbox("Facility", options=facility_labels2)
        sc_facility_id = sc_fac_label.split("(ID ", 1)[1][:-1]

        when = datetime.combine(sc_date, sc_time)

        pred_duration = predict_duration_minutes(
            models=models, when=when, facility_id=sc_facility_id, district=str(sc_district)
        )
        pred_occ = predict_occupancy(models=models, when=when)

        pred_cost = float("nan")
        if np.isfinite(pred_duration):
            pred_cost = predict_cost(models=models, duration_minutes=pred_duration, when=when)

        p1, p2, p3 = st.columns(3)
        if np.isfinite(pred_duration):
            p1.metric("Predicted duration", f"{pred_duration:,.0f} min")
        else:
            p1.metric("Predicted duration", "—")
            p1.caption("Selected facility/district not in training data.")

        p2.metric("Predicted occupancy", f"{pred_occ:,.0f} cars")
        p3.metric("Predicted cost", f"${pred_cost:,.2f}" if np.isfinite(pred_cost) else "—")

        st.divider()
        st.subheader("Drivers (model feature importance)")

        fi1, fi2, fi3 = st.columns(3)
        dur_feats = ["arrival_hour", "day_of_week", "month", "facility", "district"]
        dur_imp = pd.Series(
            models.duration_model.feature_importances_, index=dur_feats
        ).sort_values(ascending=False)
        fi1.write("**Duration model**")
        fi1.bar_chart(dur_imp)

        occ_feats = ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"]
        occ_imp = pd.Series(
            models.occupancy_model.feature_importances_, index=occ_feats
        ).sort_values(ascending=False)
        fi2.write("**Occupancy model**")
        fi2.bar_chart(occ_imp)

        fee_feats = ["duration_minutes", "duration_squared", "arrival_hour", "day_of_week"]
        fee_imp = pd.Series(models.cost_model.feature_importances_, index=fee_feats).sort_values(
            ascending=False
        )
        fi3.write("**Cost model**")
        fi3.bar_chart(fee_imp)

    with tab_forecast:
        st.subheader("Occupancy forecast")
        models = train_models(df)

        st.write("**Quick forecast (Random Forest)**")
        start_at = df["IN_TIME"].max().to_pydatetime().replace(minute=0, second=0, microsecond=0)
        rf_fc = rf_forecast_next_hours(models=models, start_at=start_at, hours=24)
        st.line_chart(rf_fc, height=260)

        st.divider()
        enable_lstm = st.toggle("Enable LSTM forecast (slower)", value=False)
        if enable_lstm:
            st.write("**LSTM forecast (next 24 hours)**")
            try:
                with st.spinner("Training LSTM forecaster (cached for this session)..."):
                    events_all = compute_events(df)
                    lstm_fc = lstm_forecast_next_hours(events_all, hours=24)
                if len(lstm_fc):
                    st.line_chart(lstm_fc, height=260)
                else:
                    st.warning("Not enough data points to train the LSTM forecaster.")
            except RuntimeError as e:
                st.error(str(e))
                st.info(
                    "Tip: Your error log shows Streamlit running from Anaconda "
                    "(`/Users/bpalan/opt/anaconda3/...`). Use the repo venv instead."
                )

    with tab_data:
        st.subheader("Filtered sessions (sample)")
        st.dataframe(
            df_f.sort_values("IN_TIME", ascending=False).head(200),
            use_container_width=True,
            height=400,
        )

        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data as CSV",
            data=csv,
            file_name="filtered_parking_sessions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()


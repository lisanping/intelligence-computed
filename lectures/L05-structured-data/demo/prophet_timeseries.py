"""
L05 – Structured Data: Time-Series Forecasting with Prophet
============================================================
Fit Facebook Prophet on the classic airline-passengers pattern and
forecast 24 months ahead.  Shows forecast + confidence intervals and
trend / seasonality component decomposition.

Usage
-----
    python prophet_timeseries.py
    python prophet_timeseries.py --periods 36
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1337
np.random.seed(SEED)
warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def generate_airline_data() -> pd.DataFrame:
    """Synthetic monthly data mimicking the classic airline-passengers pattern."""
    dates = pd.date_range("1949-01-01", periods=144, freq="MS")
    t = np.arange(len(dates))
    trend = 110 + 2.2 * t
    seasonal = 30 * np.sin(2 * np.pi * t / 12) + 12 * np.cos(4 * np.pi * t / 12)
    noise = np.random.normal(0, 8, len(t))
    passengers = trend + seasonal + noise
    return pd.DataFrame({"ds": dates, "y": passengers})


def main():
    parser = argparse.ArgumentParser(description="Prophet time-series demo")
    parser.add_argument("--periods", type=int, default=24, help="Months to forecast")
    args = parser.parse_args()

    # --- data -----------------------------------------------------------
    df = generate_airline_data()

    # --- fit Prophet ----------------------------------------------------
    from prophet import Prophet

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=args.periods, freq="MS")
    forecast = model.predict(future)

    # --- plot forecast --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["ds"], df["y"], "k.", markersize=3, label="Observed")
    ax.plot(forecast["ds"], forecast["yhat"], "b-", label="Forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"],
                    forecast["yhat_upper"], alpha=0.25, label="95 % CI")
    ax.set(xlabel="Date", ylabel="Passengers", title="Airline Passengers – Prophet Forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prophet_forecast.png"), dpi=150)
    print(f"[saved] {FIG_DIR}/prophet_forecast.png")

    # --- component decomposition ----------------------------------------
    fig2 = model.plot_components(forecast)
    fig2.savefig(os.path.join(FIG_DIR, "prophet_components.png"), dpi=150)
    print(f"[saved] {FIG_DIR}/prophet_components.png")
    plt.show()


if __name__ == "__main__":
    main()

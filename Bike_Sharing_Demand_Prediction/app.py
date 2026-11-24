import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sqlite3
import json
import ssl
import certifi
from urllib.request import urlopen
from urllib.parse import urlencode

from datetime import date, datetime
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bike Sharing Analytics",
    page_icon="🚲",
    layout="wide"
)

DB_PATH = Path(__file__).parent / "bike_predictions.db"

# ==========================================
# SQLITE DATABASE
# ==========================================

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                prediction_time TEXT,
                travel_date TEXT,
                hour INTEGER,

                temperature REAL,
                humidity INTEGER,
                wind_speed REAL,
                visibility INTEGER,

                dew_point REAL,
                solar_radiation REAL,
                rainfall REAL,
                snowfall REAL,

                season TEXT,
                holiday TEXT,
                functioning_day TEXT,

                predicted_rentals INTEGER

            )
        """)

        conn.commit()


def save_prediction(record):

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO predictions (

                prediction_time,
                travel_date,
                hour,
                temperature,
                humidity,
                wind_speed,
                visibility,
                dew_point,
                solar_radiation,
                rainfall,
                snowfall,
                season,
                holiday,
                functioning_day,
                predicted_rentals

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, record)

        conn.commit()


def get_prediction_history():

    with get_connection() as conn:

        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )


def delete_prediction(record_id):

    with get_connection() as conn:

        conn.execute(
            "DELETE FROM predictions WHERE id = ?",
            (record_id,)
        )

        conn.commit()


initialize_database()

# ==========================================
# LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():

    return joblib.load(
        "bike_sharing_random_forest.pkl"
    )


@st.cache_data
def load_data():

    data = pd.read_csv(
        "bike_sharing_data.csv"
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        dayfirst=True,
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date"]
    ).copy()

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    data["IsWeekend"] = (
        data["DayOfWeek"] >= 5
    ).astype(int)

    return data


try:

    model = load_model()
    df = load_data()

except Exception as e:

    st.error(f"Project loading error: {e}")
    st.stop()


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #00D4AA;
}

[data-testid="stMetric"] {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.stButton>button {
    background-color: #00A884;
    color: white;
    border-radius: 8px;
    border: none;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🚲 BIKE ANALYTICS")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Demand Prediction",
        "Demand Analytics",
        "Weather Analysis",
        "Model Performance",
        "Model Explainability",
        "Prediction History",
        "AI Insights",
        "Live Weather & Forecast"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Project\n\n"
    "VOIS × Edunet Foundation"
)


# ==========================================
# HOME DASHBOARD
# ==========================================

if page == "Home Dashboard":

    st.title("🚲 Bike Sharing Demand Prediction")

    st.subheader(
        "Intelligent Bike Rental Demand Analytics System"
    )

    st.write(
        "Analyze historical bike rentals and predict "
        "future demand using Random Forest Regression."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Bike Rentals",
        f"{df['Rented Bike Count'].sum():,}"
    )

    col2.metric(
        "Average Rentals",
        f"{df['Rented Bike Count'].mean():.0f}"
    )

    col3.metric(
        "Maximum Rentals",
        f"{df['Rented Bike Count'].max():,}"
    )

    col4.metric(
        "Average Temperature",
        f"{df['Temperature(°C)'].mean():.1f} °C"
    )

    st.divider()

    st.subheader("📊 Hourly Bike Rental Demand")

    hourly = (
        df.groupby("Hour")["Rented Bike Count"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="Hour",
        y="Rented Bike Count",
        markers=True,
        title="Average Bike Rentals by Hour",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌦️ Seasonal Demand")

        seasonal = (
            df.groupby("Seasons")["Rented Bike Count"]
            .mean()
            .reset_index()
        )

        fig = px.pie(
            seasonal,
            names="Seasons",
            values="Rented Bike Count",
            hole=0.45,
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("📈 Monthly Demand")

        monthly = (
            df.groupby("Month")["Rented Bike Count"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            monthly,
            x="Month",
            y="Rented Bike Count",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# DEMAND PREDICTION
# ==========================================

elif page == "Demand Prediction":

    st.title("🤖 Bike Demand Prediction")

    st.write(
        "Enter weather and time information "
        "to predict bike rental demand."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        selected_date = st.date_input(
            "Select Date",
            value=date(2018, 11, 15)
        )

        hour = st.slider(
            "Hour of Day",
            0, 23, 18
        )

        temperature = st.number_input(
            "Temperature (°C)",
            -20.0, 40.0, 20.0
        )

        humidity = st.slider(
            "Humidity (%)",
            0, 100, 50
        )

        wind_speed = st.number_input(
            "Wind Speed (m/s)",
            0.0, 20.0, 2.0
        )

        visibility = st.number_input(
            "Visibility (10m)",
            0, 2000, 1000
        )

        dew_point = st.number_input(
            "Dew Point Temperature (°C)",
            -30.0, 40.0, 10.0
        )

    with col2:

        solar = st.number_input(
            "Solar Radiation (MJ/m2)",
            0.0, 5.0, 0.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            0.0, 100.0, 0.0
        )

        snowfall = st.number_input(
            "Snowfall (cm)",
            0.0, 20.0, 0.0
        )

        season = st.selectbox(
            "Season",
            ["Spring", "Summer", "Autumn", "Winter"]
        )

        holiday = st.selectbox(
            "Holiday",
            ["No Holiday", "Holiday"]
        )

        functioning = st.selectbox(
            "Functioning Day",
            ["Yes", "No"]
        )

    st.divider()

    if st.button(
        "🚀 Predict Bike Demand",
        use_container_width=True
    ):

        prediction_date = pd.Timestamp(
            selected_date
        )

        input_data = pd.DataFrame([{

            "Hour": hour,

            "Temperature(°C)": temperature,

            "Humidity(%)": humidity,

            "Wind speed (m/s)": wind_speed,

            "Visibility (10m)": visibility,

            "Dew point temperature(°C)": dew_point,

            "Solar Radiation (MJ/m2)": solar,

            "Rainfall(mm)": rainfall,

            "Snowfall (cm)": snowfall,

            "Seasons": season,

            "Holiday": holiday,

            "Functioning Day": functioning,

            "Year": prediction_date.year,

            "Month": prediction_date.month,

            "Day": prediction_date.day,

            "DayOfWeek": prediction_date.dayofweek,

            "IsWeekend": int(
                prediction_date.dayofweek >= 5
            )

        }])

        try:

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(
                0,
                round(prediction)
            )

            st.success(
                "Bike Demand Prediction Completed!"
            )

            st.metric(
                "Predicted Bike Rentals",
                f"{prediction:,} Bikes"
            )

            # SAVE TO SQLITE DATABASE

            save_prediction((

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                str(selected_date),

                hour,

                temperature,

                humidity,

                wind_speed,

                visibility,

                dew_point,

                solar,

                rainfall,

                snowfall,

                season,

                holiday,

                functioning,

                prediction

            ))

            st.success(
                "✅ Prediction permanently saved to SQLite database!"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )


# ==========================================
# DEMAND ANALYTICS
# ==========================================

elif page == "Demand Analytics":

    st.title("📊 Demand Analytics")

    hourly = (
        df.groupby("Hour")["Rented Bike Count"]
        .mean()
        .reset_index()
    )

    st.subheader("Hourly Rental Analysis")

    fig = px.bar(
        hourly,
        x="Hour",
        y="Rented Bike Count",
        color="Rented Bike Count",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Weekday vs Weekend")

    day_data = df.copy()

    day_data["Day Type"] = np.where(
        day_data["DayOfWeek"] >= 5,
        "Weekend",
        "Weekday"
    )

    comparison = (
        day_data.groupby("Day Type")["Rented Bike Count"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        comparison,
        x="Day Type",
        y="Rented Bike Count",
        color="Day Type",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Seasonal Demand")

    seasonal = (
        df.groupby("Seasons")["Rented Bike Count"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        seasonal,
        x="Seasons",
        y="Rented Bike Count",
        color="Seasons",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# WEATHER ANALYSIS
# ==========================================

elif page == "Weather Analysis":

    st.title("🌦️ Weather Impact Analysis")

    st.subheader("Temperature vs Rental Demand")

    fig = px.scatter(
        df,
        x="Temperature(°C)",
        y="Rented Bike Count",
        color="Seasons",
        opacity=0.5,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Humidity vs Rental Demand")

    fig = px.scatter(
        df,
        x="Humidity(%)",
        y="Rented Bike Count",
        color="Seasons",
        opacity=0.5,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Rainfall Impact")

    rainfall_data = (
        df.groupby("Rainfall(mm)")["Rented Bike Count"]
        .mean()
        .reset_index()
    )

    fig = px.scatter(
        rainfall_data,
        x="Rainfall(mm)",
        y="Rented Bike Count",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("📈 Random Forest Model Performance")

    feature_columns = [

        "Hour",
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
        "Seasons",
        "Holiday",
        "Functioning Day",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "IsWeekend"

    ]

    evaluation_data = df.sort_values(
        ["Date", "Hour"]
    ).reset_index(drop=True)

    X = evaluation_data[feature_columns]

    y = evaluation_data["Rented Bike Count"]

    split_index = int(len(X) * 0.80)

    X_test = X.iloc[split_index:]

    y_test = y.iloc[split_index:]

    try:

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{mae:.2f}")

        col2.metric("RMSE", f"{rmse:.2f}")

        col3.metric("R² Score", f"{r2:.4f}")

        comparison = pd.DataFrame({

            "Actual Rentals": y_test.values,

            "Predicted Rentals": predictions

        })

        st.subheader("Actual vs Predicted")

        fig = px.scatter(
            comparison,
            x="Actual Rentals",
            y="Predicted Rentals",
            template="plotly_dark",
            opacity=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Model evaluation error: {e}"
        )


# ==========================================
# MODEL EXPLAINABILITY
# ==========================================

elif page == "Model Explainability":

    st.title("🔍 Model Explainability")
    st.write(
        "Understand which input features the trained Random Forest model uses most, "
        "and inspect how its predictions compare with historical rentals."
    )
    st.caption(
        "Feature importance describes the model's overall reliance on features. "
        "It does not prove causation or explain the direction of an individual prediction."
    )
    st.divider()

    feature_columns = [
        "Hour",
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
        "Seasons",
        "Holiday",
        "Functioning Day",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "IsWeekend"
    ]

    # Support both a standalone RandomForestRegressor and a Pipeline
    # containing preprocessing followed by the Random Forest estimator.
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model.steps[-1][1])
        fitted_preprocessor = model.named_steps.get("preprocessor")
    else:
        estimator = model
        fitted_preprocessor = None

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)

        # When preprocessing expands categorical columns (for example,
        # OneHotEncoder), use the transformed feature names so the labels
        # match the importance values exactly.
        if fitted_preprocessor is not None and hasattr(fitted_preprocessor, "get_feature_names_out"):
            try:
                importance_names = list(fitted_preprocessor.get_feature_names_out())
            except Exception:
                importance_names = []
        else:
            model_feature_names = getattr(estimator, "feature_names_in_", None)
            importance_names = list(model_feature_names) if model_feature_names is not None else []

        if len(importance_names) != len(importances):
            if len(feature_columns) == len(importances):
                importance_names = feature_columns
            else:
                importance_names = [f"Feature {i + 1}" for i in range(len(importances))]

        importance_df = pd.DataFrame({
            "Feature": importance_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.subheader("🌟 Which features matter most? ")
        top_n = st.slider(
            "Number of features to display",
            min_value=1,
            max_value=max(1, len(importance_df)),
            value=min(10, max(1, len(importance_df)))
        )
        top_features = importance_df.head(top_n).sort_values("Importance", ascending=True)
        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            importance_df.assign(Importance=importance_df["Importance"].map(lambda x: f"{x:.4f}")),
            use_container_width=True,
            hide_index=True
        )
        st.info(
            "A larger importance score means the feature contributed more to the forest's "
            "splitting decisions across the fitted trees. Correlated features can share or "
            "distort importance, so interpret rankings carefully."
        )
    else:
        st.warning(
            "The loaded model does not expose feature_importances_. "
            "Check that the saved model contains a fitted RandomForestRegressor."
        )

    st.divider()
    st.subheader("📊 Actual vs predicted rentals on a chronological holdout")
    try:
        explain_data = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
        available_features = [col for col in feature_columns if col in explain_data.columns]
        missing_features = [col for col in feature_columns if col not in explain_data.columns]

        if missing_features:
            st.warning("Dataset columns missing for evaluation: " + ", ".join(missing_features))
        elif len(explain_data) < 10:
            st.warning("Not enough dataset rows to create a reliable holdout evaluation.")
        else:
            split_at = int(len(explain_data) * 0.80)
            X_holdout = explain_data.loc[split_at:, feature_columns]
            y_holdout = explain_data.loc[split_at:, "Rented Bike Count"]
            y_hat = model.predict(X_holdout)

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("MAE", f"{mean_absolute_error(y_holdout, y_hat):.2f} bikes")
            metric2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_holdout, y_hat)):.2f} bikes")
            metric3.metric("R² score", f"{r2_score(y_holdout, y_hat):.4f}")

            comparison_df = pd.DataFrame({
                "Actual rentals": y_holdout.to_numpy(),
                "Predicted rentals": y_hat
            })
            fig = px.scatter(
                comparison_df,
                x="Actual rentals",
                y="Predicted rentals",
                opacity=0.45,
                title="Actual rentals compared with model predictions",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residual analysis helps identify whether prediction errors
            # are centered around zero and whether large errors are common.
            comparison_df["Prediction error"] = (
                comparison_df["Predicted rentals"] - comparison_df["Actual rentals"]
            ).round(1)

            st.divider()
            st.subheader("🧪 Prediction Error (Residual) Analysis")
            st.caption("Residual = predicted rentals − actual rentals. Positive values mean overprediction; negative values mean underprediction.")

            residual_col1, residual_col2 = st.columns(2)
            with residual_col1:
                residual_fig = px.histogram(
                    comparison_df, x="Prediction error", nbins=40,
                    title="Distribution of prediction errors",
                    template="plotly_dark"
                )
                st.plotly_chart(residual_fig, use_container_width=True)
            with residual_col2:
                residual_scatter = px.scatter(
                    comparison_df, x="Predicted rentals", y="Prediction error",
                    opacity=0.45, title="Prediction error vs predicted demand",
                    template="plotly_dark"
                )
                residual_scatter.add_hline(y=0, line_dash="dash")
                st.plotly_chart(residual_scatter, use_container_width=True)

            st.subheader("📥 Download Evaluation Results")
            report = comparison_df.copy()
            report.insert(0, "Test row", np.arange(1, len(report) + 1))
            metrics_report = pd.DataFrame([{
                "MAE": mean_absolute_error(y_holdout, y_hat),
                "RMSE": np.sqrt(mean_squared_error(y_holdout, y_hat)),
                "R2 Score": r2_score(y_holdout, y_hat),
                "Test Records": len(y_holdout)
            }])
            download_col1, download_col2 = st.columns(2)
            download_col1.download_button(
                "Download prediction report (CSV)",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="bike_prediction_evaluation.csv",
                mime="text/csv"
            )
            download_col2.download_button(
                "Download model metrics (CSV)",
                data=metrics_report.to_csv(index=False).encode("utf-8"),
                file_name="bike_model_metrics.csv",
                mime="text/csv"
            )

            st.subheader("Detailed prediction results")
            st.dataframe(comparison_df.head(100), use_container_width=True, hide_index=True)
            st.caption("The charts and metrics use the final 20% of rows after sorting by date and hour.")
    except Exception as e:
        st.error(f"Explainability evaluation could not be completed: {e}")


# ==========================================
# PERMANENT PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    st.title("📋 Permanent Prediction History")

    st.write(
        "All predictions are stored in your SQLite database."
    )

    history = get_prediction_history()

    if not history.empty:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predictions",
            len(history)
        )

        col2.metric(
            "Average Predicted Rentals",
            f"{history['predicted_rentals'].mean():.0f}"
        )

        col3.metric(
            "Maximum Predicted Rentals",
            f"{history['predicted_rentals'].max():,}"
        )

        st.divider()

        st.subheader("🔍 Search Prediction History")

        search_date = st.date_input(
            "Filter by travel date",
            value=None
        )

        filtered_history = history.copy()

        if search_date is not None:

            filtered_history = filtered_history[
                filtered_history["travel_date"]
                == str(search_date)
            ]

        st.dataframe(
            filtered_history,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("📥 Download History")

        csv = filtered_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction History CSV",
            data=csv,
            file_name="bike_prediction_history.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("🗑️ Delete a Prediction")

        available_ids = history["id"].tolist()

        selected_id = st.selectbox(
            "Select Prediction ID",
            available_ids
        )

        if st.button(
            "Delete Selected Prediction"
        ):

            delete_prediction(selected_id)

            st.success(
                f"Prediction ID {selected_id} deleted successfully."
            )

            st.rerun()

    else:

        st.info(
            "No prediction records found. "
            "Make your first bike demand prediction!"
        )


# ==========================================
# AI-POWERED DEMAND INSIGHTS
# ==========================================

elif page == "AI Insights":

    st.title("🧠 AI-Powered Demand Insights")
    st.write(
        "Data-driven insights and operational recommendations "
        "calculated from the historical bike-sharing dataset."
    )
    st.caption(
        "These insights describe patterns in the supplied historical data; "
        "they are not live forecasts or proof of causation."
    )
    st.divider()

    # Peak and low-demand hours from actual historical records
    hourly_insights = (
        df.groupby("Hour")["Rented Bike Count"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .sort_values("Hour")
    )

    peak_row = hourly_insights.loc[hourly_insights["mean"].idxmax()]
    low_row = hourly_insights.loc[hourly_insights["mean"].idxmin()]
    busiest_hour = int(peak_row["Hour"])
    quietest_hour = int(low_row["Hour"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak demand hour", f"{busiest_hour:02d}:00", f"{peak_row['mean']:.0f} avg. rentals")
    c2.metric("Lowest demand hour", f"{quietest_hour:02d}:00", f"{low_row['mean']:.0f} avg. rentals")
    c3.metric("Overall hourly average", f"{df['Rented Bike Count'].mean():.0f} rentals")

    st.subheader("🔥 Demand by hour")
    hourly_chart = px.bar(
        hourly_insights, x="Hour", y="mean",
        labels={"mean": "Average rentals", "Hour": "Hour of day"},
        title="Average bike rentals for each hour",
        template="plotly_dark"
    )
    hourly_chart.update_traces(
        marker_color=["#00D4AA" if int(h) == busiest_hour else "#4776A8" for h in hourly_insights["Hour"]]
    )
    st.plotly_chart(hourly_chart, use_container_width=True)

    st.subheader("🌦️ Weather impact")
    weather_col1, weather_col2 = st.columns(2)

    with weather_col1:
        # Compare records with and without recorded rainfall
        rain_comparison = df.copy()
        rain_comparison["Rain condition"] = np.where(
            rain_comparison["Rainfall(mm)"].fillna(0) > 0, "Rain", "No rain"
        )
        rain_summary = (
            rain_comparison.groupby("Rain condition")["Rented Bike Count"]
            .mean().reset_index()
        )
        rain_fig = px.bar(
            rain_summary, x="Rain condition", y="Rented Bike Count",
            color="Rain condition", title="Average rentals: rain vs no rain",
            template="plotly_dark"
        )
        st.plotly_chart(rain_fig, use_container_width=True)

    with weather_col2:
        temp_summary = (
            df.assign(Temperature_Band=pd.cut(
                df["Temperature(°C)"],
                bins=[-float("inf"), 0, 10, 20, 30, float("inf")],
                labels=["Below 0°C", "0–10°C", "10–20°C", "20–30°C", "Above 30°C"]
            ))
            .groupby("Temperature_Band", observed=False)["Rented Bike Count"]
            .mean().reset_index()
            .dropna()
        )
        temp_fig = px.bar(
            temp_summary, x="Temperature_Band", y="Rented Bike Count",
            title="Average rentals by temperature range",
            labels={"Temperature_Band": "Temperature range"},
            template="plotly_dark"
        )
        st.plotly_chart(temp_fig, use_container_width=True)

    st.subheader("🍂 Seasonal demand comparison")
    seasonal_insights = (
        df.groupby("Seasons")["Rented Bike Count"]
        .agg(["mean", "sum", "count"]).reset_index()
        .sort_values("mean", ascending=False)
    )
    if not seasonal_insights.empty:
        best_season = seasonal_insights.iloc[0]
        st.info(
            f"The highest average demand in this dataset occurs in **{best_season['Seasons']}** "
            f"({best_season['mean']:.0f} rentals per recorded hour on average)."
        )
        season_fig = px.bar(
            seasonal_insights, x="Seasons", y="mean", color="Seasons",
            labels={"mean": "Average rentals"},
            title="Average bike rentals by season", template="plotly_dark"
        )
        st.plotly_chart(season_fig, use_container_width=True)

    st.subheader("🚲 Bike allocation recommendations")
    st.write(
        "Use the historical hourly averages below as a planning baseline. "
        "The suggested buffer is a configurable planning assumption, not a model prediction."
    )
    buffer_pct = st.slider("Extra availability buffer (%)", 0, 50, 15, 5)
    allocation = hourly_insights[["Hour", "mean"]].copy()
    allocation["Recommended bikes"] = np.ceil(
        allocation["mean"] * (1 + buffer_pct / 100)
    ).astype(int)
    allocation = allocation.rename(columns={"Hour": "Hour of day", "mean": "Historical average rentals"})
    allocation["Historical average rentals"] = allocation["Historical average rentals"].round(1)
    allocation["Hour of day"] = allocation["Hour of day"].map(lambda x: f"{int(x):02d}:00")
    st.dataframe(allocation, use_container_width=True, hide_index=True)

    st.subheader("💡 Automatically generated findings")
    findings = []
    findings.append(
        f"Peak historical demand is at {busiest_hour:02d}:00, averaging "
        f"{peak_row['mean']:.0f} rentals per recorded hour."
    )
    findings.append(
        f"The quietest recorded hour is {quietest_hour:02d}:00, averaging "
        f"{low_row['mean']:.0f} rentals."
    )
    if len(rain_summary) == 2:
        rain_means = rain_summary.set_index("Rain condition")["Rented Bike Count"]
        difference = rain_means.get("No rain", np.nan) - rain_means.get("Rain", np.nan)
        if pd.notna(difference):
            if difference > 0:
                findings.append(f"No-rain records average {difference:.1f} more rentals than rainy records.")
            elif difference < 0:
                findings.append(f"Rainy records average {abs(difference):.1f} more rentals than no-rain records; check sample sizes before acting on this pattern.")
            else:
                findings.append("Average rentals are similar in rainy and no-rain records.")
    if not seasonal_insights.empty:
        findings.append(
            f"{seasonal_insights.iloc[0]['Seasons']} has the highest seasonal average, "
            f"while {seasonal_insights.iloc[-1]['Seasons']} has the lowest."
        )
    for item in findings:
        st.markdown(f"- {item}")

    st.download_button(
        "📥 Download hourly allocation plan",
        data=allocation.to_csv(index=False).encode("utf-8"),
        file_name="bike_hourly_allocation_plan.csv",
        mime="text/csv"
    )


# ==========================================
# LIVE WEATHER & FORECAST (SEOUL)
# ==========================================

elif page == "Live Weather & Forecast":

    st.title("🌦️ Live Weather & Bike Demand Forecast")
    st.write(
        "Fetch current conditions and the next 24 hours of weather for Seoul, "
        "then estimate hourly bike demand with your saved Random Forest model."
    )
    st.info(
        "The training dataset represents Seoul. This page therefore uses Seoul coordinates "
        "(37.5665° N, 126.9780° E) to keep weather geography consistent. "
        "Weather forecasts are external API data; demand estimates are model outputs."
    )

    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_seoul_weather():
        params = {
            "latitude": 37.5665,
            "longitude": 126.9780,
            "timezone": "Asia/Seoul",
            "forecast_days": 2,
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "current": (
                "temperature_2m,relative_humidity_2m,dew_point_2m,"
                "apparent_temperature,is_day,precipitation,rain,snowfall,"
                "weather_code,cloud_cover,pressure_msl,wind_speed_10m,"
                "wind_direction_10m,visibility"
            ),
            "hourly": (
                "temperature_2m,relative_humidity_2m,dew_point_2m,"
                "precipitation_probability,precipitation,rain,snowfall,"
                "weather_code,cloud_cover,wind_speed_10m,visibility,"
                "shortwave_radiation"
            )
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
        # Use the up-to-date CA bundle provided by certifi to avoid expired
        # or outdated local certificate stores on some Windows installations.
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(url, timeout=20, context=ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))

    def weather_label(code):
        labels = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
            53: "Moderate drizzle", 55: "Dense drizzle", 56: "Freezing drizzle",
            57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain",
            65: "Heavy rain", 66: "Freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
        }
        return labels.get(int(code), "Unknown conditions")

    def season_from_month(month):
        # Seasons used by the Seoul training dataset.
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        if month in (9, 10, 11):
            return "Autumn"
        return "Winter"

    def predict_weather_hour(hour_row, holiday_value, functioning_value):
        stamp = pd.Timestamp(hour_row["time"])
        # Open-Meteo visibility is in metres; the training feature is in 10-metre units.
        visibility_10m = float(hour_row.get("visibility", 10000) or 10000) / 10.0
        # Open-Meteo shortwave radiation is W/m²; convert to hourly MJ/m².
        solar_mj = float(hour_row.get("shortwave_radiation", 0) or 0) * 0.0036
        features = {
            "Hour": int(stamp.hour),
            "Temperature(°C)": float(hour_row.get("temperature_2m", 0) or 0),
            "Humidity(%)": int(round(float(hour_row.get("relative_humidity_2m", 0) or 0))),
            "Wind speed (m/s)": float(hour_row.get("wind_speed_10m", 0) or 0),
            "Visibility (10m)": visibility_10m,
            "Dew point temperature(°C)": float(hour_row.get("dew_point_2m", 0) or 0),
            "Solar Radiation (MJ/m2)": solar_mj,
            "Rainfall(mm)": float(hour_row.get("rain", 0) or 0),
            "Snowfall (cm)": float(hour_row.get("snowfall", 0) or 0),
            "Seasons": season_from_month(stamp.month),
            "Holiday": holiday_value,
            "Functioning Day": functioning_value,
            "Year": int(stamp.year),
            "Month": int(stamp.month),
            "Day": int(stamp.day),
            "DayOfWeek": int(stamp.dayofweek),
            "IsWeekend": int(stamp.dayofweek >= 5)
        }
        expected = list(getattr(model, "feature_names_in_", features.keys()))
        row = pd.DataFrame([{name: features[name] for name in expected}])
        return max(0, int(round(float(model.predict(row)[0]))))

    if st.button("🔄 Fetch latest Seoul weather", type="primary"):
        st.cache_data.clear()

    try:
        with st.spinner("Fetching current weather and forecast for Seoul..."):
            weather_data = fetch_seoul_weather()

        current = weather_data.get("current", {})
        current_units = weather_data.get("current_units", {})
        current_code = current.get("weather_code", 0)

        st.subheader("Current conditions in Seoul")
        st.caption(f"Local observation time: {current.get('time', 'Not available')} (Asia/Seoul)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
        k2.metric("Humidity", f"{current.get('relative_humidity_2m', '—')} %")
        k3.metric("Wind speed", f"{current.get('wind_speed_10m', '—')} m/s")
        k4.metric("Visibility", f"{(float(current.get('visibility', 0) or 0) / 1000):.1f} km")
        st.write(f"**Conditions:** {weather_label(current_code)}")
        st.write(
            f"**Dew point:** {current.get('dew_point_2m', '—')} °C  |  "
            f"**Rain:** {current.get('rain', '—')} mm  |  "
            f"**Snowfall:** {current.get('snowfall', '—')} cm  |  "
            f"**Cloud cover:** {current.get('cloud_cover', '—')} %"
        )

        hourly = weather_data.get("hourly", {})
        hourly_df = pd.DataFrame(hourly)
        if hourly_df.empty or "time" not in hourly_df:
            st.warning("The weather service did not return hourly forecast data.")
        else:
            hourly_df["time"] = pd.to_datetime(hourly_df["time"])
            now_local = pd.Timestamp(current.get("time", hourly_df["time"].iloc[0]))
            upcoming = hourly_df[hourly_df["time"] >= now_local].head(24).copy()

            st.divider()
            st.subheader("Next 24 hours: weather forecast")
            weather_chart = px.line(
                upcoming, x="time", y="temperature_2m", markers=True,
                labels={"time": "Seoul local time", "temperature_2m": "Temperature (°C)"},
                template="plotly_dark", title="Hourly temperature forecast"
            )
            st.plotly_chart(weather_chart, use_container_width=True)

            holiday_choice = st.selectbox(
                "Holiday assumption for demand estimates",
                ["No Holiday", "Holiday"], index=0,
                help="The weather API does not identify the Seoul public-holiday calendar. Choose the assumption used by the model."
            )
            functioning_choice = st.selectbox(
                "Bike service operating?", ["Yes", "No"], index=0
            )
            st.caption(
                "Demand estimates use the trained model and forecast weather. "
                "Holiday status is your selected assumption; the model does not determine actual fleet availability."
            )

            if st.button("🤖 Predict demand for the next 24 hours"):
                try:
                    if functioning_choice == "No":
                        predicted_values = [0] * len(upcoming)
                    else:
                        predicted_values = [
                            predict_weather_hour(row, holiday_choice, functioning_choice)
                            for _, row in upcoming.iterrows()
                        ]
                    result = upcoming[["time", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "rain", "snowfall"]].copy()
                    result["Predicted bike rentals"] = predicted_values
                    result = result.rename(columns={
                        "time": "Seoul local time",
                        "temperature_2m": "Temperature (°C)",
                        "relative_humidity_2m": "Humidity (%)",
                        "wind_speed_10m": "Wind speed (m/s)",
                        "rain": "Rain (mm)",
                        "snowfall": "Snowfall (cm)"
                    })
                    st.session_state["live_weather_demand"] = result
                    st.success("24-hour demand estimates generated using your Random Forest model.")
                except Exception as model_error:
                    st.error(f"Could not generate weather-based demand estimates: {model_error}")

            if "live_weather_demand" in st.session_state:
                result = st.session_state["live_weather_demand"]
                st.subheader("🚲 Predicted bike demand by hour")
                demand_fig = px.bar(
                    result, x="Seoul local time", y="Predicted bike rentals",
                    color="Predicted bike rentals", template="plotly_dark",
                    title="Random Forest demand estimates for the next 24 hours"
                )
                st.plotly_chart(demand_fig, use_container_width=True)
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download 24-hour forecast CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="seoul_24_hour_bike_demand_forecast.csv",
                    mime="text/csv"
                )

        st.caption("Weather source: Open-Meteo Forecast API. Data availability and forecast accuracy depend on the provider.")

    except Exception as weather_error:
        st.error(f"Unable to retrieve live weather right now: {weather_error}")
        st.info("Check your internet connection and try again. Your existing dashboard and prediction pages can still be used.")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center>Bike Sharing Demand Prediction | "
    "Random Forest Machine Learning | "
    "VOIS × Edunet Foundation</center>",
    unsafe_allow_html=True
)
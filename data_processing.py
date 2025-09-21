# Import Libraries

# Data handling
import pandas as pd
import numpy as np

# Geospatial & datetime utilities
from sklearn.metrics.pairwise import haversine_distances  # compute nearest site by lat/lon
from datetime import timedelta                           # select tickets within last 24 hours

# Preprocessing
from sklearn.preprocessing import LabelEncoder           # encode categorical features: weather, antenna_type, site_id, root_cause
from sklearn.preprocessing import OneHotEncoder          # encode: event, alarm
from sklearn.model_selection import train_test_split     # split train/test datasets

# ML pipeline: scaling + SMOTE + model
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Model evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

# Model export / save
import joblib

# Database
import sqlite3


# Connect to DB
conn = sqlite3.connect("database/vip_server.db")

# Load datasets from SQL tables into pandas DataFrames
kpi_df = pd.read_sql("SELECT * FROM kpi_timeseries", conn)
alarms_df = pd.read_sql("SELECT * FROM alarms", conn)
tickets_df = pd.read_sql("SELECT * FROM tickets", conn)
topology_df = pd.read_sql("SELECT * FROM topology", conn)
events_df = pd.read_sql("SELECT * FROM events", conn)
weather_df = pd.read_sql("SELECT * FROM weather", conn)

conn.close()


# Set Proper Datetime Format
# Convert timestamp/date columns to datetime dtype
kpi_df["timestamp"] = pd.to_datetime(kpi_df["timestamp"])
alarms_df["timestamp"] = pd.to_datetime(alarms_df["timestamp"])
tickets_df["timestamp"] = pd.to_datetime(tickets_df["timestamp"])
events_df["date"] = pd.to_datetime(events_df["date"])
weather_df["date"] = pd.to_datetime(weather_df["date"])


# 💧 Fill Missing Values with Median
# Replace NaN values in all numerical columns of each dataset with the column median

# 1️⃣ KPI timeseries
kpi_numeric_cols = kpi_df.select_dtypes(include=np.number).columns
kpi_df[kpi_numeric_cols] = kpi_df[kpi_numeric_cols].fillna(kpi_df[kpi_numeric_cols].median())

# 2️⃣ Alarms
alarms_numeric_cols = alarms_df.select_dtypes(include=np.number).columns
alarms_df[alarms_numeric_cols] = alarms_df[alarms_numeric_cols].fillna(alarms_df[alarms_numeric_cols].median())

# 3️⃣ Tickets / complaints
tickets_numeric_cols = tickets_df.select_dtypes(include=np.number).columns
tickets_df[tickets_numeric_cols] = tickets_df[tickets_numeric_cols].fillna(tickets_df[tickets_numeric_cols].median())

# 4️⃣ Network topology
topology_numeric_cols = topology_df.select_dtypes(include=np.number).columns
topology_df[topology_numeric_cols] = topology_df[topology_numeric_cols].fillna(topology_df[topology_numeric_cols].median())

# 5️⃣ Events / external factors
events_numeric_cols = events_df.select_dtypes(include=np.number).columns
events_df[events_numeric_cols] = events_df[events_numeric_cols].fillna(events_df[events_numeric_cols].median())

# 6️⃣ Weather data
weather_numeric_cols = weather_df.select_dtypes(include=np.number).columns
weather_df[weather_numeric_cols] = weather_df[weather_numeric_cols].fillna(weather_df[weather_numeric_cols].median())


# ❌ Drop Duplicate Rows
# Remove duplicate entries from all datasets

kpi_df = kpi_df.drop_duplicates()
alarms_df = alarms_df.drop_duplicates()
tickets_df = tickets_df.drop_duplicates()
topology_df = topology_df.drop_duplicates()
events_df = events_df.drop_duplicates()
weather_df = weather_df.drop_duplicates()


# Merge on 'site_id' to enrich KPI records with site metadata
features = pd.merge(kpi_df, topology_df, on="site_id", how="left")

# 🔄 Encode Categorical Column: antenna_type
le_antenna_type = LabelEncoder()
features["antenna_type"] = le_antenna_type.fit_transform(features["antenna_type"])

# 💾 Save LabelEncoder for future use
joblib.dump(le_antenna_type, "encoder/le_antenna_type.pkl")


# Create a new 'date' column from timestamp and ensure dtype is datetime
features["date"] = pd.to_datetime(features["timestamp"].dt.date)

# 🔢 One-Hot Encode Event Types
# Convert categorical 'type' column in events_df into one-hot encoded columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
events_encoded = ohe.fit_transform(events_df[["type"]])

# Create a dataframe from one-hot results
events_one_hot = pd.DataFrame(events_encoded, columns=ohe.get_feature_names_out(["type"]))

# 💾 Save LabelEncoder for future use
joblib.dump(ohe, "ohe_events.pkl")

# Combine one-hot encoded columns with 'date' and 'site_id'
events_df_2 = pd.concat([events_df[["date", "site_id"]], events_one_hot], axis=1)

# 🔄 Aggregate Events by Site and Date
# If a site has multiple events on the same day, take the max (1 if any event occurred)
events_df_2 = events_df_2.groupby(["date", "site_id"]).max().reset_index()

# 🔗 Merge Aggregated Events into Features
# Merge on 'date' and 'site_id' to add event information to each KPI record
features = pd.merge(features, events_df_2, on=["date", "site_id"], how="left")

# 🔍 Check Missing Values for Event Columns
events_types = ["type_maintenance", "type_concert", "type_government"]

# 💧 Fill Missing Event Columns with 0
# Replace NaN values in one-hot encoded event columns with 0
# Convert dtype to integer
features[events_types] = features[events_types].fillna(0).astype(int)


# 🌦️ Function to Find Nearest Weather by Lat/Lon
def find_nearest_weather(lat, lon, date, weather_df):
    """
    Finds the nearest weather record for a given site and date.

    Parameters:
    - lat, lon: float, site coordinates
    - date: datetime, date of interest
    - weather_df: DataFrame containing weather data with 'lat', 'lon', 'date'

    Returns:
    - Series of the nearest weather record, or None if no data for that date
    """

    # Filter weather data for the same date
    same_day = weather_df[weather_df["date"] == date]
    if same_day.empty:
        return None  # Return None if no weather data available on that date

    # Convert degrees to radians (haversine_distances expects radians)
    site_long_lat = np.radians([[lat, lon]])
    weather_long_lat = np.radians(same_day[["lat", "lon"]].values)

    # Compute distance between site and weather stations (in km)
    # haversine_distances output is in radians, multiply by Earth's radius (6371 km)
    distances = haversine_distances(site_long_lat, weather_long_lat) * 6371

    # Find the index of the nearest weather station
    nearest_index = distances[0].argmin()
    return same_day.iloc[nearest_index]


# 🌦️ Apply find_nearest_weather Function to Features

# Create an empty list to store nearest weather records for each site
nearest_records = []

# Loop through each row in features dataframe
for _, row in features.iterrows():
    # Find nearest weather record for the site and date
    nearest_weather = find_nearest_weather(row["lat"], row["lon"], row["date"], weather_df)

    # If a weather record exists for that date
    if nearest_weather is not None:
        # Append a dictionary with relevant weather info
        nearest_records.append({
            "site_id": row["site_id"],
            "date": row["date"],
            "rain_mm": nearest_weather["rain_mm"],
            "temperature_c": nearest_weather["temperature_c"],
            "weather": nearest_weather["weather"]
        })

# Convert the list of nearest weather records into a DataFrame
nearest_weather_df = pd.DataFrame(nearest_records)

# ❌ Drop Duplicate Weather Records
# Keep only one record per 'site_id' and 'date'
nearest_weather_df = nearest_weather_df.drop_duplicates(subset=["site_id", "date"])

# 🔄 Encode Categorical Column: weather
# Convert 'weather' from string/object to numeric using Label Encoding
le_weather = LabelEncoder()
nearest_weather_df["weather"] = le_weather.fit_transform(nearest_weather_df["weather"])

# 💾 Save LabelEncoder for future use
joblib.dump(le_weather, "encoder/le_weather.pkl")

# 🔗 Merge Nearest Weather Data into Features
# Merge on 'site_id' and 'date' to add weather info to each KPI record
features = pd.merge(features, nearest_weather_df, on=["site_id", "date"], how="left")

# ❌ Drop Unnecessary Column
# Remove 'date' column as it's no longer needed for modeling
features = features.drop("date", axis=1)


# 🔄 Encode Categorical Column: severity
# Convert 'severity' from string/object to numeric using Label Encoding
le_severity_label = LabelEncoder()
alarms_df["severity_label"] = le_severity_label.fit_transform(alarms_df["severity"])

# 🕒 Sort Alarms Data
# Sort by 'timestamp' and 'site_id' for proper time-series aggregation
alarms_df = alarms_df.sort_values(["timestamp", "site_id"])

# ⏱️ Set Timestamp as Index
# Prepare alarms dataframe for rolling time window aggregation
alarms_df.set_index("timestamp", inplace=True)

# 🔢 Create Dummy Columns for Alarm Severity
# Convert severity labels into separate binary columns for aggregation
alarms_df["alarm_critical"] = (alarms_df["severity_label"] == 3).astype(int)
alarms_df["alarm_major"] = (alarms_df["severity_label"] == 2).astype(int)
alarms_df["alarm_minor"] = (alarms_df["severity_label"] == 1).astype(int)
alarms_df["alarm_total"] = 1  # Count every alarm

# ⏱️ Rolling 3-Hour Alarm Aggregation per Site
alarm_numeric_columns = ["alarm_total", "alarm_critical", "alarm_major", "alarm_minor"]

# Compute rolling sum of alarms over a 3-hour window for each site
alarms_df_rolling = (
    alarms_df[alarm_numeric_columns]
    .groupby(alarms_df["site_id"])
    .rolling("3h")
    .sum()
    .reset_index()
)

# ❌ Drop Duplicate Rows in Rolling Alarms DataFrame
# Keep only one record per 'site_id' and 'timestamp'
alarms_df_rolling = alarms_df_rolling.drop_duplicates(subset=["site_id", "timestamp"])

# 🔗 Merge Rolling Alarms into Features
# Merge on 'site_id' and 'timestamp' to add 3-hour rolling alarm info to each KPI record
features = pd.merge(features, alarms_df_rolling, on=["site_id", "timestamp"], how="left")

# 🔄 Convert Alarm Columns to Integer and Fill Missing Values
# Replace NaN with 0 and ensure dtype is int for all alarm count columns
alarm_cols = ["alarm_total", "alarm_critical", "alarm_major", "alarm_minor"]
features[alarm_cols] = features[alarm_cols].fillna(0).astype(int)


# 📝 Group Tickets by Site
# Create a dictionary with site_id as key and a list of (timestamp, category_label) tuples as value
tickets_grouped = tickets_df.groupby("site_id")[["timestamp", "category_label"]].apply(
    lambda x: list(x.itertuples(index=False, name=None))
).to_dict()

# 🕒 Function to Check for Tickets Within 24 Hours
def check_ticket(row):
    site = row["site_id"]
    ts = row["timestamp"]

    # If site has no tickets, return 'No' and placeholder category
    if site not in tickets_grouped:
        return pd.Series({"label_24h": "No", "category_label": "-"})

    # Loop through all tickets for the site
    for tiket, kategori in tickets_grouped[site]:
        # If a ticket occurs after the current timestamp but within the next 24 hours → return 'Yes'
        if ts < tiket <= ts + timedelta(hours=24):
            return pd.Series({"label_24h": "Yes", "category_label": kategori})

    # If no ticket occurs within 24 hours
    return pd.Series({"label_24h": "No", "category_label": "-"})

# Apply the function to create target columns
features[["label_24h", "category_label"]] = features.apply(check_ticket, axis=1)


# 🔄 Encode 'site_id' Column
# Convert site_id from string/object to numeric using Label Encoding
le_site_id = LabelEncoder()
features["site_id"] = le_site_id.fit_transform(features["site_id"])

# 💾 Save LabelEncoder for future use
joblib.dump(le_site_id, "encoder/le_site_id.pkl")

# 🔄 Encode Binary Target: label_24h
# Map 'No' to 0 and 'Yes' to 1
features["label_24h"] = features["label_24h"].map({"No": 0, "Yes": 1})

# 🔄 Encode Multiclass Target: category_label
# Convert category_label from string/object to numeric using Label Encoding
le_category_label = LabelEncoder()
features["category_label"] = le_category_label.fit_transform(features["category_label"])

# 💾 Save LabelEncoder for future use
joblib.dump(le_category_label, "encoder/le_category_label.pkl")


# 💾 Export Processed Features to CSV
# Save the fully prepared features dataframe for modeling
features.to_csv("features.csv", index=False)


# 🎯 Split Features (X) and Targets (y)
# Drop 'timestamp' because models require numeric input
X = features.drop(["timestamp", "label_24h", "category_label"], axis=1)

# Level 1 target (binary)
y1 = features["label_24h"]

# Level 2 target (multiclass)
y2 = features["category_label"]


# 🔀 Split Dataset into Training and Test Sets

# Level 1 target (label_24h)
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y1, test_size=0.2, random_state=42, stratify=y1
)

# Level 2 target (category_label)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y2, test_size=0.2, random_state=42, stratify=y2
)


# Load Level 1 model (Binary)
model_level1 = joblib.load("models/level1/XGB_Level1.pkl")

# Load Level 2 model (Multiclass)
model_level2 = joblib.load("models/level2/Random_Forest_Level2.pkl")

# Load saved label encoders
le_site_id = joblib.load("encoder/le_site_id.pkl")
le_category_label = joblib.load("encoder/le_category_label.pkl")
le_antenna_type = joblib.load("encoder/le_antenna_type.pkl")
le_weather = joblib.load("encoder/le_weather.pkl")
ohe_events = joblib.load("encoder/ohe_events.pkl")


def level3_recommendation(row):
    """
    Rule-based mapping of KPI thresholds to Root Cause recommendations.

    Input:
        row : pd.Series, a single row of features (KPI values)
    Output:
        Tuple(root_cause_category, sub_root_cause, recommended_action)
    """

    # ---------------- Tier 1: Availability ----------------
    if row["availability"] < 98:
        return "Availability", f"Availability Problem ({int(row['availability'])}%)", "Troubleshoot"

    # ---------------- Tier 1: Quality Alarms ----------------
    elif row["cpri_alarm"] == 1:
        return "Quality", "Trans Alarm CPRI", "Troubleshoot"
    elif row["vswr_alarm"] == 1:
        return "Quality", "Trans Alarm VSWR", "Troubleshoot"

    # ---------------- Tier 2: Quality Troubleshoot ----------------
    elif row["ul_interference_db"] > -100:
        return "Quality", f"High UL Interference ({int(row['ul_interference_db'])} dB)", "Troubleshoot"
    elif row["packet_loss_pct"] > 0.1:
        return "Quality", f"Trans Packet Loss ({int(row['packet_loss_pct']*100)}%)", "Troubleshoot"

    # ---------------- Tier 3: Coverage ----------------
    elif row["rsrp"] < -105:
        return "Coverage", f"Low RSRP ({int(row['rsrp'])} dBm)", "Optimization"
    elif row["rsrq"] < -15:
        return "Coverage", f"Low RSRQ ({int(row['rsrq'])} dB)", "Optimization"
    elif row["sinr"] < 13:
        return "Coverage", f"Low SINR ({int(row['sinr'])} dB)", "Optimization"

    # ---------------- Tier 4: Capacity ----------------
    elif row["prb_dl_util_pct"] > 90:
        return "Capacity", f"High PRB DL ({int(row['prb_dl_util_pct'])}%)", "Optimization"
    elif row["prb_ul_util_pct"] > 90:
        return "Capacity", f"High PRB UL ({int(row['prb_ul_util_pct'])}%)", "Optimization"
    elif row["active_user"] > 20:
        return "Capacity", f"High Active User ({int(row['active_user'])})", "Optimization"
    elif row["max_user"] > 150:
        return "Capacity", f"High Max User ({int(row['max_user'])})", "Optimization"

    # ---------------- Tier 5: Quality Optimization ----------------
    elif row["latency_ms"] > 20:
        return "Quality", f"High Latency ({int(row['latency_ms'])} ms)", "Optimization"
    elif row["cssr"] < 98:
        return "Quality", f"Performance Problem ({int(row['cssr'])}%)", "Optimization"

    # ---------------- Tier 6: Service ----------------
    elif row["wa_success_ratio"] < 98:
        return "Service", f"Low WA Success Ratio ({int(row['wa_success_ratio'])}%)", "Check Service / Application"
    elif row["volte_drop_rate"] > 0.1:
        return "Service", f"High VoLTE Drop Rate ({row['volte_drop_rate']}%)", "Check Service / Application"
    elif row["sms_success_ratio"] < 98:
        return "Service", f"Low SMS Success Ratio ({int(row['sms_success_ratio'])}%)", "Check Service / Application"
    elif row["gaming_latency_ms"] > 20:
        return "Service", f"High Gaming Latency ({int(row['gaming_latency_ms'])} ms)", "Check Service / Application"
    elif row["throughput_dl_mbps"] < 20:
        return "Service", f"Low Throughput DL ({int(row['throughput_dl_mbps'])} Mbps)", "Check Service / Application"

    # ---------------- All Normal ----------------
    return "Others", "Network Normal", "No action needed"


# 📌 Predict New Data: Levels 1, 2, and 3

results = []

for index, row in X.iterrows():
    # Level 1: Predict if site will have a complaint in next 24h
    pred_24h = model_level1.predict([row])[0]

    if pred_24h == 1:
        # Level 2: Predict complaint category (multiclass)
        pred_category = model_level2.predict([row])[0]

        # Level 3: Rule-based root cause recommendation
        root_cause, sub_root_cause, action = level3_recommendation(row)

    else:
        # No complaint predicted → set Level 2 & 3 as NaN
        pred_category, root_cause, sub_root_cause, action = np.nan, np.nan, np.nan, np.nan

    # Collect results into a list
    results.append({
        "timestamp": features.loc[row.name, "timestamp"],
        "site_id": row["site_id"],
        "sector": row["sector"].astype(int),
        "latitude": row["lat"],
        "longitude": row["lon"],
        "pred_24h": pred_24h,
        "pred_category": pred_category,
        "root_cause": root_cause,
        "sub_root_cause": sub_root_cause,
        "action": action,
        "azimuth": row["azimuth_A"].astype(int),
        "tilt": row["tilt_A"].astype(int),
        "antenna_type": row["antenna_type"],
        "band": row["band"].astype(int),
        # event
        "rain_mm": row["rain_mm"].astype(int),
        "temperature_c": row["temperature_c"].astype(int),
        "weather": row["weather"],
        # alarm
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Decode categorical features back to original labels

# site_id
results_df["site_id"] = le_site_id.inverse_transform(results_df["site_id"].astype(int))
# pred_24h (binary Yes/No)
results_df["pred_24h"] = results_df["pred_24h"].map({0: "No", 1: "Yes"})
# pred_category (multiclass)
def safe_inverse_label(x, le):
    if pd.isna(x):
        return np.nan
    try:
        return le.inverse_transform([int(x)])[0]
    except ValueError:
        return np.nan  # unseen label jadi NaN

results_df["pred_category"] = results_df["pred_category"].apply(lambda x: safe_inverse_label(x, le_category_label))
# antenna_type
results_df["antenna_type"] = le_antenna_type.inverse_transform(results_df["antenna_type"].astype(int))
# weather
results_df["weather"] = le_weather.inverse_transform(results_df["weather"].astype(int))
# event_type
event_cols = [c for c in X.columns if c.startswith("type_")]
if event_cols:
    decoded_type = []
    for i, row_ev in X[event_cols].iterrows():
        try:
            decoded = ohe_events.inverse_transform([row_ev.values])[0][0]  # ambil 1 label
        except IndexError:  # jika semua 0 (ga ada event)
            decoded = np.nan
        decoded_type.append(decoded)
    results_df["event_type"] = decoded_type

# to Streamlit
def get_prediction_results():
    return results_df

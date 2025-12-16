import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.impute import SimpleImputer  # NEW: handle NaNs [web:102]

from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NSL-KDD RF with Upload & Distribution",
    layout="wide"
)

st.title("NSL-KDD Intrusion Detection with Random Forest")
st.write(
    "Train a Random Forest on NSL-KDD (with SMOTE, feature selection, and imputation), "
    "and analyze the distribution of attacks in uploaded data."
)

# ------------------------------------------------------------------
# Common metadata
# ------------------------------------------------------------------
NSL_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

ATTACK_CATEGORIES = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
    'udpstorm': 'DoS', 'worm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data
def load_data(train_file: str, test_file: str):
    df_train = pd.read_csv(train_file, names=NSL_COLUMNS)
    df_test = pd.read_csv(test_file, names=NSL_COLUMNS)
    return df_train, df_test


# ------------------------------------------------------------------
# Preprocessing + feature selection + IMPUTATION
# ------------------------------------------------------------------
@st.cache_data
def preprocess_data(df_train: pd.DataFrame,
                    df_test: pd.DataFrame,
                    top_k_features: int):
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Map attack types to categories
    df_train['attack_category'] = df_train['attack_type'].map(ATTACK_CATEGORIES)
    df_test['attack_category'] = df_test['attack_type'].map(ATTACK_CATEGORIES)

    df_train['attack_category'].fillna('Unknown', inplace=True)
    df_test['attack_category'].fillna('Unknown', inplace=True)

    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        le_dict[col] = le

    # Encode target
    label_encoder = LabelEncoder()
    df_train['attack_label'] = label_encoder.fit_transform(df_train['attack_category'])
    df_test['attack_label'] = label_encoder.transform(df_test['attack_category'])

    # Features/labels
    X_train_full = df_train.drop(
        ['attack_type', 'attack_category', 'attack_label', 'difficulty'], axis=1
    )
    y_train = df_train['attack_label']

    X_test_full = df_test.drop(
        ['attack_type', 'attack_category', 'attack_label', 'difficulty'], axis=1
    )
    y_test = df_test['attack_label']

    # Scale using all original features first (no imputation here, just for feature ranking)
    scaler_full = StandardScaler()
    X_train_scaled_full = scaler_full.fit_transform(X_train_full)
    X_test_scaled_full = scaler_full.transform(X_test_full)

    # Quick RF for feature importance (no SMOTE here, just to rank features)
    base_rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    base_rf.fit(X_train_scaled_full, y_train)

    feature_importance = pd.DataFrame({
        'feature': X_train_full.columns,
        'importance': base_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Select top-k features
    if top_k_features > len(feature_importance):
        top_k_features = len(feature_importance)

    selected_features = feature_importance['feature'].head(top_k_features).tolist()

    # Rebuild X with selected features only
    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    # IMPUTE missing values before scaling and SMOTE [web:98][web:111]
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale after imputation
    scaler_selected = StandardScaler()
    X_train_scaled = scaler_selected.fit_transform(X_train_imputed)
    X_test_scaled = scaler_selected.transform(X_test_imputed)

    return (X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            label_encoder,
            df_train,
            df_test,
            selected_features,
            feature_importance,
            scaler_selected,
            le_dict,
            imputer)


# ------------------------------------------------------------------
# Model training with SMOTE
# ------------------------------------------------------------------
@st.cache_resource
def train_model_with_smote(X_train_scaled,
                           y_train,
                           n_estimators: int,
                           max_depth,
                           use_class_weight: bool):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

    cw = "balanced" if use_class_weight else None

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=cw,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_res, y_res)
    return rf_model


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    # Sidebar: training settings
    st.sidebar.header("Training Settings")

    train_file = st.sidebar.text_input("Train file path", "KDDTrain+.txt")
    test_file = st.sidebar.text_input("Test file path", "KDDTest+.txt")

    n_estimators = st.sidebar.slider("n_estimators (trees)", 50, 500, 200, 50)
    max_depth_opt = st.sidebar.selectbox("max_depth", ["None", 10, 20, 40])
    max_depth = None if max_depth_opt == "None" else int(max_depth_opt)

    top_k_features = st.sidebar.slider("Top-k features (feature selection)",
                                       5, 41, 20, 1)
    use_class_weight = st.sidebar.checkbox("Use class_weight='balanced'",
                                           value=True)

    run_btn = st.sidebar.button("Run Training & Evaluation")

    # ------------------------------------------------------------------
    # Training section
    # ------------------------------------------------------------------
    rf_model = None
    scaler_selected = None
    label_encoder = None
    selected_features = None
    le_dict = None
    imputer = None

    if run_btn:
        with st.spinner("Loading data..."):
            df_train, df_test = load_data(train_file, test_file)

        with st.spinner("Preprocessing, imputing & selecting features..."):
            (X_train_scaled,
             X_test_scaled,
             y_train,
             y_test,
             label_encoder,
             df_train_proc,
             df_test_proc,
             selected_features,
             feature_importance,
             scaler_selected,
             le_dict,
             imputer) = preprocess_data(
                df_train,
                df_test,
                top_k_features
            )

        st.subheader("Attack Category Distribution (Train)")
        col1, col2 = st.columns(2)

        with col1:
            st.write(df_train_proc['attack_category'].value_counts())

        with col2:
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            df_train_proc['attack_category'].value_counts().plot(kind='bar', ax=ax_dist)
            ax_dist.set_title("Attack Category Distribution (Train)")
            ax_dist.set_xlabel("Attack Category")
            ax_dist.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig_dist)

        st.subheader("Selected Features")
        st.write(f"Top {len(selected_features)} features used for training:")
        st.write(selected_features)

        st.subheader("Feature Importance (All Features)")
        st.dataframe(feature_importance.head(20))

        fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=feature_importance.head(20),
            x="importance",
            y="feature",
            ax=ax_fi
        )
        ax_fi.set_title("Top 20 Features by Importance (Base RF)")
        st.pyplot(fig_fi)

        with st.spinner("Training Random Forest with SMOTE..."):
            rf_model = train_model_with_smote(
                X_train_scaled,
                y_train,
                n_estimators,
                max_depth,
                use_class_weight
            )

        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")

        st.text("Classification Report:")
        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            digits=4
        )
        st.text(report)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax_cm
        )
        ax_cm.set_title("Confusion Matrix - Attack Category Classification")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

        st.success("Training and evaluation complete.")

    # ------------------------------------------------------------------
    # Uploaded Data Analysis
    # ------------------------------------------------------------------
    st.header("Uploaded Data Analysis")

    st.write(
        "Upload an NSL-KDD style file (same 43 columns). "
        "If it has `attack_type`, the app will show the **true** attack category "
        "distribution. If not, and a model is trained, it will show the "
        "**predicted** attack category distribution (with imputation)."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV/TXT (NSL-KDD format, no header)",
        type=["csv", "txt"]
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, names=NSL_COLUMNS)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        st.subheader("Uploaded Data Preview")
        st.write(df_upload.head())

        has_attack_type = 'attack_type' in df_upload.columns and df_upload['attack_type'].notnull().any()

        # Case 1: uploaded file has true labels
        if has_attack_type:
            df_upload['attack_category'] = df_upload['attack_type'].map(ATTACK_CATEGORIES)
            df_upload['attack_category'].fillna('Unknown', inplace=True)

            st.subheader("True Attack Category Distribution (Uploaded Data)")
            st.write(df_upload['attack_category'].value_counts())

            fig_up, ax_up = plt.subplots(figsize=(6, 4))
            df_upload['attack_category'].value_counts().plot(kind='bar', ax=ax_up)
            ax_up.set_title("True Attack Category Distribution (Uploaded Data)")
            ax_up.set_xlabel("Attack Category")
            ax_up.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig_up)

        # Case 2: unlabeled file, predict only if model is trained
        else:
            if (
                rf_model is None or scaler_selected is None or
                label_encoder is None or selected_features is None or
                le_dict is None or imputer is None
            ):
                st.warning("Train the model first (left sidebar) to enable prediction on unlabeled uploads.")
            else:
                st.subheader("Predicted Attack Category Distribution (Uploaded Data)")

                df_pred = df_upload.copy()

                # Encode categorical columns using fitted LabelEncoders
                for col in ['protocol_type', 'service', 'flag']:
                    if col in df_pred.columns:
                        classes = list(le_dict[col].classes_)
                        mapping = {v: i for i, v in enumerate(classes)}
                        df_pred[col] = df_pred[col].map(lambda x: mapping.get(x, 0))

                # Drop non-feature columns
                for c in ['attack_type', 'difficulty']:
                    if c in df_pred.columns:
                        df_pred = df_pred.drop(columns=[c])

                missing = [f for f in selected_features if f not in df_pred.columns]
                if missing:
                    st.error(f"Uploaded data is missing expected feature columns: {missing}")
                else:
                    X_upload = df_pred[selected_features]

                    # Impute using the same imputer as training [web:98][web:110]
                    X_upload_imputed = imputer.transform(X_upload)

                    # Scale
                    X_upload_scaled = scaler_selected.transform(X_upload_imputed)

                    # Predict
                    y_upload_pred = rf_model.predict(X_upload_scaled)
                    cat_pred = label_encoder.inverse_transform(y_upload_pred)

                    df_pred_result = df_upload.copy()
                    df_pred_result['predicted_category'] = cat_pred

                    st.write(df_pred_result.head())

                    st.write(df_pred_result['predicted_category'].value_counts())

                    fig_up2, ax_up2 = plt.subplots(figsize=(6, 4))
                    df_pred_result['predicted_category'].value_counts().plot(kind='bar', ax=ax_up2)
                    ax_up2.set_title("Predicted Attack Category Distribution (Uploaded Data)")
                    ax_up2.set_xlabel("Attack Category")
                    ax_up2.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_up2)


if __name__ == "__main__":
    main()

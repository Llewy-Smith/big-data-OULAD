#### pipelines.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Config:
    DATA_DIR = Path("Datasets/Open University Learning Analytics Dataset (OULAD)")
    EARLY_CUTOFF_PCT = 0.25
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    FAILURE_THRESHOLD = 40
    EXPORT_CSVS = True

CSV_FILES = {
    "courses": "courses.csv",
    "assessments": "assessments.csv",
    "vle": "vle.csv",
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "studentAssessment": "studentAssessment.csv",
    "studentVLE": "studentVLE.csv",
}

def load_raw_data(verbose=False):
    if verbose:
        print("Loading data...")
    dfs = {}
    for name, filename in CSV_FILES.items():
        filepath = Config.DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot find {filepath}")
        df = pd.read_csv(filepath)
        dfs[name] = df
        if verbose:
            print(f" {name}: {df.shape}")
    return dfs

def add_course_cutoffs(courses_df):
    courses = courses_df.copy()
    courses["early_cutoff"] = Config.EARLY_CUTOFF_PCT * courses["module_presentation_length"]
    return courses

def create_vle_features(studentVLE_df, courses_df):
    vle = studentVLE_df.merge(
        courses_df[["code_module", "code_presentation", "early_cutoff"]],
        on=["code_module", "code_presentation"]
    )
    vle = vle[vle["date"] <= vle["early_cutoff"]]
    vle_agg = vle.groupby(["id_student", "code_module", "code_presentation"]).agg(
        total_clicks=("sum_click", "sum"),
        days_active=("date", "nunique"),
        last_login_date=("date", "max")
    ).reset_index()
    vle_agg = vle_agg.merge(
        courses_df[["code_module", "code_presentation", "early_cutoff"]],
        on=["code_module", "code_presentation"]
    )
    vle_agg["days_since_last_login"] = vle_agg["early_cutoff"] - vle_agg["last_login_date"]
    vle_agg = vle_agg.drop(["early_cutoff"], axis=1)
    return vle_agg

def create_assessment_features(studentAssessment_df, assessments_df, courses_df):
    df = studentAssessment_df.merge(
        assessments_df[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]],
        on="id_assessment"
    ).merge(
        courses_df[["code_module", "code_presentation", "early_cutoff"]],
        on=["code_module", "code_presentation"]
    )
    df = df[(df["assessment_type"] != "Exam") & (df["date_submitted"] <= df["early_cutoff"]) & (df["score"].notna())]
    if df.empty:
        return pd.DataFrame(columns=["id_student", "code_module", "code_presentation", "early_avg_score", "early_failed_any"])
    df["failed"] = (df["score"] < Config.FAILURE_THRESHOLD).astype(int)
    df["weighted_score"] = df["score"] * df["weight"]
    agg = df.groupby(["id_student", "code_module", "code_presentation"]).agg(
        total_weighted_score=("weighted_score", "sum"),
        total_weight=("weight", "sum"),
        failed_any=("failed", "max")
    ).reset_index()
    agg["early_avg_score"] = agg["total_weighted_score"] / agg["total_weight"]
    agg["early_failed_any"] = agg["failed_any"]
    return agg[["id_student", "code_module", "code_presentation", "early_avg_score", "early_failed_any"]]

def merge_all_data(studentInfo_df, studentRegistration_df, courses_df, vle_features, assessment_features):
    df = studentInfo_df.merge(studentRegistration_df, on=["id_student", "code_module", "code_presentation"], how="left")
    df = df.merge(courses_df[["code_module", "code_presentation", "early_cutoff"]], on=["code_module", "code_presentation"], how="left")
    df = df.merge(vle_features, on=["id_student", "code_module", "code_presentation"], how="left")
    df = df.merge(assessment_features, on=["id_student", "code_module", "code_presentation"], how="left")
    df = df[(df["date_unregistration"].isna()) | (df["date_unregistration"] >= df["early_cutoff"])]
    return df

def create_target_variable(df):
    df = df.copy()
    df["target"] = df["final_result"].map({
        "Pass": 0, "Distinction": 0,
        "Withdrawn": 1, "Fail": 1
    })
    return df

def encode_categorical_variables(df):
    df = df.copy()
    df["is_male"] = df["gender"].map({"M": 1, "F": 0})
    df["has_disability"] = df["disability"].map({"Y": 1, "N": 0})
    df["age_numeric"] = df["age_band"].map({"0-35": 0, "35-55": 1, "55<=": 2})
    df["education_numeric"] = df["highest_education"].map({
        "No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
        "HE Qualification": 3, "Post Graduate Qualification": 4
    })
    df["imd_numeric"] = df["imd_band"].map({
        "0-10%": 0, "10-20%": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
        "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9
    })
    return df

def handle_missing_values(df):
    df = df.copy()
    df["total_clicks"] = df["total_clicks"].fillna(0)
    df["days_active"] = df["days_active"].fillna(0)
    df["early_avg_score"] = df["early_avg_score"].fillna(0)
    df["early_failed_any"] = df["early_failed_any"].fillna(0)
    df["days_since_last_login"] = df["days_since_last_login"].fillna(df["early_cutoff"].max())
    df["imd_numeric"] = df["imd_numeric"].fillna(9)
    return df

def select_final_features(df):
    feature_cols = [
        "target", "is_male", "age_numeric", "education_numeric", "has_disability",
        "imd_numeric", "studied_credits", "num_of_prev_attempts", "code_module",
        "total_clicks", "days_active", "days_since_last_login",
        "early_avg_score", "early_failed_any", "date_registration"
    ]
    df_final = df[feature_cols].copy()
    df_final = df_final.rename(columns={"date_registration": "days_enrolled_before_course"})
    return df_final.dropna(subset=["target", "days_enrolled_before_course"])

def split_train_test(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y)

def add_course_difficulty(X_train, X_test, y_train):
    course_difficulty = X_train.copy()
    course_difficulty["target"] = y_train
    course_map = course_difficulty.groupby("code_module")["target"].mean()
    fallback = course_map.mean()
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    X_train_out["course_difficulty"] = X_train["code_module"].map(course_map).fillna(fallback)
    X_test_out["course_difficulty"] = X_test["code_module"].map(course_map).fillna(fallback)
    return X_train_out, X_test_out

def add_relative_assessment_performance(X_train, X_test):
    medians = X_train.groupby("code_module")["early_avg_score"].median()
    fallback = medians.mean()
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    for X in [X_train_out, X_test_out]:
        X["relative_assessment_performance"] = X["early_avg_score"] - X["code_module"].map(medians).fillna(fallback)
    return X_train_out.drop("early_avg_score", axis=1), X_test_out.drop("early_avg_score", axis=1)

def add_relative_vle_engagement(X_train, X_test):
    medians = X_train.groupby("code_module").agg({
        "total_clicks": "median", "days_active": "median"
    })
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    for X in [X_train_out, X_test_out]:
        X["relative_total_clicks"] = X["total_clicks"] - X["code_module"].map(medians["total_clicks"])
        X["relative_days_active"] = X["days_active"] - X["code_module"].map(medians["days_active"])
    return X_train_out.drop(["total_clicks", "days_active"], axis=1), X_test_out.drop(["total_clicks", "days_active"], axis=1)

def remove_course_identifier(X_train, X_test):
    return X_train.drop("code_module", axis=1), X_test.drop("code_module", axis=1)

def scale_features(X_train, X_test):
    numeric = [
        "studied_credits", "num_of_prev_attempts", "course_difficulty",
        "relative_assessment_performance", "relative_total_clicks", "relative_days_active",
        "days_since_last_login", "days_enrolled_before_course",
        "age_numeric", "education_numeric", "imd_numeric"
    ]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric] = scaler.fit_transform(X_train[numeric])
    X_test_scaled[numeric] = scaler.transform(X_test[numeric])
    return X_train_scaled, X_test_scaled, scaler

def prepare_data_for_modeling(scaled=True, verbose=False):
    if verbose:
        print("="*60)
    dfs = load_raw_data(verbose=verbose)
    courses = add_course_cutoffs(dfs["courses"])
    vle = create_vle_features(dfs["studentVLE"], courses)
    assess = create_assessment_features(dfs["studentAssessment"], dfs["assessments"], courses)
    merged = merge_all_data(dfs["studentInfo"], dfs["studentRegistration"], courses, vle, assess)
    merged = create_target_variable(merged)
    merged = encode_categorical_variables(merged)
    merged = handle_missing_values(merged)
    final_df = select_final_features(merged)
    X_train, X_test, y_train, y_test = split_train_test(final_df)
    X_train, X_test = add_course_difficulty(X_train, X_test, y_train)
    X_train, X_test = add_relative_assessment_performance(X_train, X_test)
    X_train, X_test = add_relative_vle_engagement(X_train, X_test)
    X_train, X_test = remove_course_identifier(X_train, X_test)
    if scaled:
        X_train_final, X_test_final, scaler = scale_features(X_train, X_test)
        if Config.EXPORT_CSVS:
            pd.concat([X_train_final, y_train], axis=1).to_csv("train_data.csv", index=False)
            pd.concat([X_test_final, y_test], axis=1).to_csv("test_data.csv", index=False)
    else:
        X_train_final, X_test_final, scaler = X_train, X_test, None
        if Config.EXPORT_CSVS:
            pd.concat([X_train, y_train], axis=1).to_csv("train_data_raw.csv", index=False)
            pd.concat([X_test, y_test], axis=1).to_csv("test_data_raw.csv", index=False)
    features = X_train_final.columns.tolist()
    return X_train_final, X_test_final, y_train, y_test, scaler, features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = prepare_data_for_modeling()
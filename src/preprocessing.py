import pandas as pd


def load_raw_data(path="data/raw/diabetic_data.csv"):
    """
    Load the raw CSV WITHOUT letting pandas guess missing values.

    Why: by default pandas turns the text "None" into NaN. In this dataset
    "None" is a real value for max_glu_serum / A1Cresult ("test not done"),
    and DECISIONS D5 maps it to "Not measured". keep_default_na=False keeps
    every value as the original text, so our own rules do the cleaning.
    """
    return pd.read_csv(path, keep_default_na=False)


# Broad ICD-9 diagnosis categories
ICD9_RANGES = [
    (1, 139, "Infectious"),
    (140, 239, "Neoplasms"),
    (240, 279, "Endocrine_Metabolic"),
    (280, 289, "Blood"),
    (290, 319, "Mental"),
    (320, 389, "Nervous"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital"),
    (760, 779, "Perinatal"),
    (780, 799, "Symptoms_Signs"),
    (800, 999, "Injury_Poisoning"),
]


def map_diagnosis(code):
    """
    Convert an ICD-9 diagnosis code into a broad category.
    """

    code = str(code).strip()

    if code == "?":
        return "Unknown"

    if code.startswith("V"):
        return "V-code"

    if code.startswith("E"):
        return "E-code"

    try:
        numeric_code = int(float(code))
    except ValueError:
        return "Unknown"

    for lower, upper, category in ICD9_RANGES:
        if lower <= numeric_code <= upper:
            return category

    return "Unknown"


def add_diagnosis_categories(df):
    """
    Add broad categorical versions of diag_1, diag_2 and diag_3.
    """

    df = df.copy()

    for column in ["diag_1", "diag_2", "diag_3"]:
        df[f"{column}_category"] = df[column].apply(map_diagnosis)

    return df

def handle_missing_categories(df):
    """
    Replace dataset-specific missing-value markers
    with meaningful categorical labels.
    """

    df = df.copy()

    unknown_columns = [
        "race",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    not_measured_columns = [
        "max_glu_serum",
        "A1Cresult",
    ]

    for column in unknown_columns:
        df[column] = df[column].replace("?", "Unknown")

    for column in not_measured_columns:
        df[column] = df[column].replace("None", "Not measured")

    return df

def prepare_dataset(df):
    """
    Apply the project's finalized data-preparation decisions.

    Returns:
        X      : model features
        y      : binary target
        groups : patient IDs used only for grouped splitting
    """

    df = df.copy()

    # 1. Remove encounters with explicit expired dispositions
    df = df[~df["discharge_disposition_id"].isin([11, 19, 20])].copy()

    # 2. Create binary target
    # <30 days = 1
    # NO or >30 days = 0
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # 3. Keep patient_nbr separately for grouped train/test splitting
    groups = df["patient_nbr"].copy()

    # 4. Handle dataset-specific missing values
    df = handle_missing_categories(df)

    # 5. Create broad diagnosis categories
    df = add_diagnosis_categories(df)

    # 6. Remove columns that must not be model features
    columns_to_drop = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "examide",
        "citoglipton",
        "readmitted",
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    X = df.drop(columns=columns_to_drop + ["target"])
    y = df["target"]

    return X, y, groups

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
]


def create_preprocessor(exclude=()):
    """
    Create the preprocessing pipeline for numerical
    and categorical features.

    exclude: optional feature names to leave out (used for the fairness
    ablation, e.g. exclude=("race", "gender")). Default keeps all 44.
    """
    numeric = [c for c in NUMERIC_FEATURES if c not in exclude]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in exclude]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                categorical,
            ),
        ]
    )

    return preprocessor
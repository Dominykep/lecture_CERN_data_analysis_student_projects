import pandas as pd
import numpy as np

def remove_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:

    # print('Removing completely empty rows and columns...')
    # Work on a copy so you don't mutate original by accident
    df_clean = df.copy()
    df_clean = df_clean.replace(r'^\s*$', np.nan, regex=True)

    # what to remove
    while True:
        choice = input("Remove (r)ows, (c)olumns, or (b)oth? [r/c/b]: ").strip().lower()
        if choice in ("r", "c", "b"):
            break
        print("Please enter r, c, or b.")

    # threshold
    while True:
        try:
            p = float(input("Enter MAX allowed % of missing values (0–100): "))
            if 0 <= p <= 100:
                break
            else:
                print("Enter a number from 0 to 100.")
        except ValueError:
            print("Invalid number.")

    threshold = p / 100.0

    # compute on ORIGINAL df_clean (before any dropping)
    row_missing_fraction = df_clean.isna().mean(axis=1)
    col_missing_fraction = df_clean.isna().mean(axis=0)

    rows_ok = row_missing_fraction <= threshold
    cols_ok = col_missing_fraction <= threshold

    if choice == "r":
        # only rows filtered
        return df_clean.loc[rows_ok, :]
    elif choice == "c":
        # only cols filtered
        return df_clean.loc[:, cols_ok]
    else:  # "b"
        # both filtered, using original percentages
        return df_clean.loc[rows_ok, cols_ok]

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    # print('Stripping whitespace from column names and cell values...')
   
    # Work on a copy for safety
    df_clean = df.copy()

    # 1) Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()

    # 2) Strip whitespace inside cells (only object/string columns)
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean

def normalize_missing_values(df):

    df_clean = df.copy()

# Add more patterns as needed   
    missing_patterns = [
        r"^\s*$",     # empty / whitespace
        r"(?i)^na$",  
        r"(?i)^n/a$",
        r"(?i)^null$",
        r"(?i)^none$",
        r"^\?$",
        r"^-$",
        r"^\.$",
    ]

    for pattern in missing_patterns:
        df_clean = df_clean.replace(pattern, pd.NA, regex=True)

    return df_clean

def fix_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == "object":  
            # Replace comma-decimal only if the value looks like a number
            df_clean[col] = (
                df_clean[col]
                .str.replace(r"(?<=\d),(?=\d)", ".", regex=True)  # 1,25 -> 1.25
            )

            # Convert to numeric where possible
            df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

    return df_clean

def extract_numeric_and_unit(df):
    
    import re

    df_clean = df.copy()

    # number (with . or ,) + optional space + unit letters
    pattern = re.compile(r"^\s*([0-9]+[.,]?[0-9]*)\s*([A-Za-zµ°%]+)\s*$")

    new_columns_order = []

    for col in df_clean.columns:
        # Only try to split object/string columns
        if df_clean[col].dtype == "object":
            series = df_clean[col].astype(str)

            # Extract number + unit into two columns (0 = number, 1 = unit)
            extracted = series.str.extract(pattern)

            # If this column actually matches the pattern at least once,
            # we treat it as numeric+unit and replace it
            if not extracted.isna().all().all():
                num_col = f"{col}_value"
                unit_col = f"{col}_unit"

                # Convert number: change comma to dot, then to float
                nums = extracted[0].str.replace(",", ".", regex=False)
                df_clean[num_col] = pd.to_numeric(nums, errors="coerce")
                df_clean[unit_col] = extracted[1]

                # Put these two where the original column was
                new_columns_order.extend([num_col, unit_col])
                continue  # skip adding the original column name

        # If not object type or no match: keep original column as-is
        new_columns_order.append(col)

    # Reorder to reflect replacements
    df_clean = df_clean[new_columns_order]

    return df_clean

def convert_units_to_SI(df: pd.DataFrame) -> pd.DataFrame:

    import math

    df_clean = df.copy()

    def _convert_one(value, unit):
        if pd.isna(value) or pd.isna(unit):
            return value, unit

        try:
            v = float(value)
        except (TypeError, ValueError):
            return value, unit

        u = str(unit).strip()
        u = u.replace("°", "deg")  # normalize degrees
        u = u.replace("µ", "u")    # normalize micro
        u = u.lower()

        # ---- temperature -> K ----
        if u in ("degc", "c"):
            return v + 273.15, "K"
        if u in ("degf", "f"):
            return (v - 32.0) * 5.0 / 9.0 + 273.15, "K"
        if u in ("k", "degk"):
            return v, "K"

        # ---- length -> m ----
        length_units = {
            "mm": 1e-3,
            "cm": 1e-2,
            "m": 1.0,
            "km": 1e3,
        }
        if u in length_units:
            return v * length_units[u], "m"

        # ---- mass -> kg ----
        mass_units = {
            "mg": 1e-6,
            "g": 1e-3,
            "kg": 1.0,
            "t": 1e3,
        }
        if u in mass_units:
            return v * mass_units[u], "kg"

        # ---- time -> s ----
        time_units = {
            "ms": 1e-3,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
        }
        if u in time_units:
            return v * time_units[u], "s"

        # ---- pressure -> Pa ----
        pressure_units = {
            "pa": 1.0,
            "kpa": 1e3,
            "mpa": 1e6,
            "bar": 1e5,
            "mbar": 1e2,
            "atm": 101325.0,
            "psi": 6894.757,
        }
        if u in pressure_units:
            return v * pressure_units[u], "Pa"

        # ---- force -> N ----
        force_units = {
            "n": 1.0,
            "kn": 1e3,
        }
        if u in force_units:
            return v * force_units[u], "N"

        # ---- energy -> J ----
        energy_units = {
            "j": 1.0,
            "kj": 1e3,
        }
        if u in energy_units:
            return v * energy_units[u], "J"

        # ---- percentage -> fraction (0–1) ----
        if u in ("%", "pct"):
            return v / 100.0, "1"   # dimensionless

        # unknown unit -> leave as is
        return value, unit

    # Look for <base>_value + <base>_unit pairs
    cols = list(df_clean.columns)
    for col in cols:
        if col.endswith("_value"):
            base = col[:-6]  # remove "_value"
            unit_col = base + "_unit"
            if unit_col in df_clean.columns:
                for idx, (val, unit) in df_clean[[col, unit_col]].iterrows():
                    new_val, new_unit = _convert_one(val, unit)
                    df_clean.at[idx, col] = new_val
                    df_clean.at[idx, unit_col] = new_unit

                # make sure numeric column is numeric dtype
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean

def remove_duplicate_rows(df):
    df_clean = df.copy()

    # Detect duplicates anywhere in the file (not only next to each other)
    dup_mask = df_clean.duplicated(keep="first")

    # Remove ALL duplicate rows, keep only the first appearance
    df_clean = df_clean[~dup_mask]

    return df_clean

def _move_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    while True:
        n_rows = len(df_clean)
        print(f"\nCurrent number of rows: {n_rows}")
        if n_rows == 0:
            print("No rows to move.")
            break

        try:
            src = int(input(f"Which row do you want to move? [1–{n_rows}]: "))
            dest = int(input(f"To which position do you want to move it? [1–{n_rows}]: "))
        except ValueError:
            print("Please enter valid integers.")
            continue

        if not (1 <= src <= n_rows and 1 <= dest <= n_rows):
            print("Row numbers out of range.")
            continue

        # build new order (squeezing, not deleting)
        indices = list(range(n_rows))
        row = indices.pop(src - 1)
        indices.insert(dest - 1, row)

        df_clean = df_clean.iloc[indices].reset_index(drop=True)

        another = input("Move another row? (yes/no): ").strip().lower()
        if another not in ("yes", "y"):
            break

    return df_clean

def _move_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    while True:
        cols = list(df_clean.columns)
        n_cols = len(cols)

        print("\nCurrent columns order:")
        for i, c in enumerate(cols, 1):
            print(f"  {i}) {c}")

        if n_cols == 0:
            print("No columns to move.")
            break

        try:
            src = int(input(f"Which column do you want to move? [1–{n_cols}]: "))
            dest = int(input(f"To which position do you want to move it? [1–{n_cols}]: "))
        except ValueError:
            print("Please enter valid integers.")
            continue

        if not (1 <= src <= n_cols and 1 <= dest <= n_cols):
            print("Column numbers out of range.")
            continue

        col_name = cols.pop(src - 1)
        cols.insert(dest - 1, col_name)

        df_clean = df_clean[cols]

        another = input("Move another column? (yes/no): ").strip().lower()
        if another not in ("yes", "y"):
            break

    return df_clean

def move_rows_or_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Let the user choose if they want to move rows or columns.
    After each move it asks if that is all – if yes, it returns
    the modified DataFrame (then your main script proceeds to saving).
    """
    df_clean = df.copy()

    while True:
        choice = input("Move (r)ows or (c)olumns? [r/c]: ").strip().lower()
        if choice in ("r", "c"):
            break
        print("Please enter 'r' for rows or 'c' for columns.")

    if choice == "r":
        return _move_rows(df_clean)
    else:
        return _move_columns(df_clean)
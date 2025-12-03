import os
import pandas as pd
from pandas.errors import ParserError

Root_dir = os.getcwd()

def browse_and_choose_csv():

    start_dir = os.getcwd()
    current_dir = start_dir

    while True:
        print("\n====================================")
        print("Current location:", current_dir)
        print("====================================")

        try:
            entries = os.listdir(current_dir)
        except PermissionError:
            print("No permission for this folder, going back.")
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                print("Already at the top level, cannot go back further.")
            else:
                current_dir = parent
            continue

        dirs = sorted([e for e in entries if os.path.isdir(os.path.join(current_dir, e))])
        csvs = sorted([e for e in entries if e.lower().endswith(".csv")])

        items = [("dir", d) for d in dirs] + [("csv", f) for f in csvs]

        if not items:
            print("(This folder is empty.)")

        # List entries
        for idx, (kind, name) in enumerate(items, 1):
            label = "[DIR]" if kind == "dir" else "[CSV]"
            print(f"{idx}) {label} {name}")

        # Commands info
        print("\nCommands:")
        print("  <number>  -> open folder / select CSV")
        print("  back      -> go back")
        print("  quit      -> exit without selecting a file")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "quit":
            return None

        if choice == "back":
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                print("Already at the top level.")
            else:
                current_dir = parent
            continue

        if not choice.isdigit():
            print("Please enter a number, 'back', or 'quit'.")
            continue

        n = int(choice)
        if not (1 <= n <= len(items)):
            print("Invalid number.")
            continue

        kind, name = items[n - 1]
        path = os.path.join(current_dir, name)

        if kind == "dir":
            current_dir = path
        else:  # CSV selected
            print(f"\nSelected CSV: {path}")
            return path
    
def choose_csv_actions():

    actions = [
        "Remove rows and columns",
        "Strip whitespace",
        "Normalize missing values",
        "Fix decimal commas",
        "Extract numeric value + units",
        "Convert units to SI",
        "Clear duplicate rows",
        "Move rows or columns",
    ]

    while True:
        print("\n==============================")
        print("Available actions:")
        print("==============================")

        for i, action in enumerate(actions, 1):
            print(f"{i}) {action}")

        print("\nCommands:")
        print("  numbers separated by space  -> select actions")
        print("  quit                        -> cancel")

        # user input like: 1 3 5
        raw = input("\nChoose actions: ").strip().lower()

        if raw == "quit":
            return None

        # split input into tokens
        tokens = raw.split()
        if not all(t.isdigit() for t in tokens):
            print("Please enter valid numbers.")
            continue

        nums = list(map(int, tokens))
        if not all(1 <= n <= len(actions) for n in nums):
            print("Some numbers were invalid.")
            continue

        # Deduplicate & map numbers to action names
        selected = [actions[n - 1] for n in dict.fromkeys(nums)]

        print("\nYou have chosen to perform:")
        for act in selected:
            print(" -", act)

        confirm = input("\nAre you sure? (yes/no): ").strip().lower()

        if confirm in ("yes", "y"):
            return selected
        else:
            print("\nOkay, let's choose again.")

def choose_output_path() -> str:
   
    print("\n==============================")
    print("Saving cleaned CSV")
    print("==============================")
    print("Project directory:", Root_dir)

    print("\nEnter output file path relative to the Project folder.")
    print("Examples:")
    print("   cleaned.csv")
    print("   Cleaned_csv/Cleaned.csv")
    print("   output/run1/cleaned.csv\n")

    while True:
        out = input("Output file path: ").strip()

        if not out.lower().endswith(".csv"):
            print("Output must end with .csv")
            continue

        # Save relative to the project folder
        output_path = os.path.join(Root_dir, out)

        # Make folder(s) if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        return output_path
    
# Helper to read messy
def load_csv_loose(csv_path: str) -> pd.DataFrame:
    """
    Tries to read CSV normally.
    If it fails with ParserError, retries with relaxed options
    and skips bad lines.
    """
    try:
        return pd.read_csv(csv_path)
    except ParserError as e:
        print("\nParserError while reading CSV:")
        print(e)
        print("\nRetrying with relaxed settings (engine='python', on_bad_lines='skip')...\n")
        return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

def main():
    csv_path = browse_and_choose_csv()
    if csv_path is None:
        print("\nNo CSV selected.")
        return

    file = load_csv_loose(csv_path)

    chosen_actions = choose_csv_actions()
    if chosen_actions is None:
        print("No actions selected.")
        return

    # ----------------------------
    # Importing actions
    # ----------------------------
    for action in chosen_actions:
        if action == "Remove rows and columns":
            from csv_actions import remove_rows_and_columns
            file = remove_rows_and_columns(file)
        if action == "Strip whitespace":
            from csv_actions import strip_whitespace
            file = strip_whitespace(file)
        if action == "Normalize missing values":
            from csv_actions import normalize_missing_values
            file = normalize_missing_values(file)
        if action == "Fix decimal commas":
            from csv_actions import fix_decimal_commas
            file = fix_decimal_commas(file)
        if action == "Extract numeric value + units":
            from csv_actions import extract_numeric_and_unit
            file = extract_numeric_and_unit(file)
        if action == "Convert units to SI":
            from csv_actions import convert_units_to_SI
            file = convert_units_to_SI(file)
        if action == "Clear duplicate rows":
            from csv_actions import remove_duplicate_rows
            file = remove_duplicate_rows(file)
        if action == "Move rows or columns":
            from csv_actions import move_rows_or_columns
            file = move_rows_or_columns(file)

        # Add more actions here like:
        # if action == "Strip whitespace":
        #     from actions import strip_whitespace
        #     file = strip_whitespace(df)
    # ----------------------------

    output_path = choose_output_path()
    file.to_csv(output_path, index=False)
    print(f"\nSaved cleaned file as:\n{output_path}")

# Needed for the program to run when executed directly and not when imported
if __name__ == "__main__":
    main()
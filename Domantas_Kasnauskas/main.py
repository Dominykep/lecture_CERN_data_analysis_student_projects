import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


#kainu valymo funkcija
def clean_price(value):
    if isinstance(value, str):
        value = value.replace("€", "").replace(" ", "").replace(",", "")
    try:
        return float(value)
    except:
        return np.nan

#plotu valymo funkcija
def clean_area(value):
    if isinstance(value, str):
        value = value.replace("m²", "").replace("m2", "").replace(",", ".").strip()
    try:
        return float(value)
    except:
        return np.nan


#isskirciu pasalinimo funkcija
def iqr_bounds(series):
    s = series.dropna()
    if s.empty:
        return 0, 0
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper


def plot_scatter(group):
    #generuoja nuomos kv m kaina / pirkimo kv m kaina
    print("Generating Chart 1")

    OFFSET_X = 0.02
    OFFSET_Y = 0.01

    #paruosiamos spalvos
    hoods = group.index.tolist()
    n_hoods = len(hoods)
    cmap = plt.cm.get_cmap("tab20", n_hoods)
    plt.figure(figsize=(12, 8))
    color_map = {hood: i for i, hood in enumerate(hoods)}
    colors = [cmap(color_map[hood]) for hood in group.index]
    plt.scatter(group["avg_purchase_sqm"], group["avg_rent_sqm"], s=100, c=colors, alpha=0.7)

    for hood, row in group.iterrows():
        plt.text(
            row["avg_purchase_sqm"] + OFFSET_X,
            row["avg_rent_sqm"] + OFFSET_Y,
            hood,
            fontsize=8,
            ha='left',
            va='bottom'
        )

    plt.xlabel("Average Purchase Price per m² (€)")
    plt.ylabel("Average Rent Price per m² (€)")
    plt.title("Average Rent vs Purchase Prices per m² by Neighbourhood")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_yield(group):
    #nuomos ir pirkimo kvadrato kainos santykio grafikas
    print("Generating Chart 2")

    top_yields = group.sort_values(by="rent_to_purchase_ratio", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_yields.index, top_yields["rent_to_purchase_ratio"], color='teal')
    plt.xlabel("Rent-to-Purchase Ratio (Avg. Monthly Rent / Avg. Purchase Price per m²)")
    plt.title("Top 10 Neighbourhoods by Estimated Monthly Rental Yield")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_counts(group):
    #nuomos ir pardavimu skelbimu kiekis rajonuose
    print("Generating Chart 3")

    top_counts = group.sort_values(by="total_listings", ascending=False).head(10)

    x = np.arange(len(top_counts.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, top_counts["n_purchase"], width, label='Purchase Listings', color='cornflowerblue')
    ax.bar(x + width / 2, top_counts["n_rent"], width, label='Rental Listings', color='darkorange')

    ax.set_ylabel('Number of Listings (After Outlier Removal)')
    ax.set_xlabel('Neighbourhood')
    ax.set_title('Top 10 Neighbourhoods: Rental vs. Purchase Listing Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(top_counts.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


LEGEND_POSITIONS = {
    'top left': 'upper left',
    'top right': 'upper right',
    'bottom left': 'lower left',
    'bottom right': 'lower right',
}


def plot_custom(data_frame, chart_type, x_column, y_column, legend_position):
    #vartotojo nustatytas grafikas
    OFFSET_X = 0.02 * (data_frame[x_column].max() - data_frame[x_column].min()) if x_column != 'hood' else 0.02
    OFFSET_Y = 0.01 * (data_frame[y_column].max() - data_frame[y_column].min())

    mpl_position = LEGEND_POSITIONS.get(legend_position.lower(), 'best')

    print(f"Generating Custom Chart")

    if x_column.lower() == 'hood':
        x_data = data_frame.index
        x_label = 'Neighbourhood'
    else:
        if x_column not in data_frame.columns:
            print(f"Error: X-axis column '{x_column}' not found.")
            return
        x_data = data_frame[x_column]
        x_label = x_column

    if y_column not in data_frame.columns:
        print(f"Error: Y-axis column '{y_column}' not found.")
        return
    y_data = data_frame[y_column]

    fig, ax = plt.subplots(figsize=(12, 8))  # Increased size for better label visibility

    try:
        if chart_type.lower() == 'scatter':
            ax.scatter(x_data, y_data, label=f'{y_column} vs {x_label}', alpha=0.7)
            ax.set_title(f"Scatter Plot: {y_column.capitalize()} vs {x_label.capitalize()} by Neighbourhood")

            #pridedamos duomenu antrastes
            for hood, row in data_frame.iterrows():
                x_val = row[x_column] if x_column != 'hood' else x_data[hood]
                y_val = row[y_column]
                if not pd.isna(x_val) and not pd.isna(y_val):
                    ax.text(
                        x_val + OFFSET_X,
                        y_val + OFFSET_Y,
                        hood,
                        fontsize=8,
                        ha='left',
                        va='bottom'
                    )

        elif chart_type.lower() == 'bar':
            if x_column.lower() != 'hood':
                x_data = data_frame.index
                x_label = 'Neighbourhood'
                print("\nNote: For a Bar Chart, the X-axis is fixed to 'Neighbourhood'.")

            ax.bar(x_data, y_data, label=y_column, color='darkgreen')
            ax.set_title(f"Bar Chart: {y_column.capitalize()} by {x_label.capitalize()}")

        ax.set_xlabel(x_label.capitalize())
        ax.set_ylabel(y_column.capitalize())
        ax.legend(loc=mpl_position)

        if x_column.lower() == 'hood':
            plt.xticks(rotation=45, ha="right")

        plt.grid(True, axis='both' if chart_type.lower() == 'scatter' else 'y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred during custom plotting: {e}")


def get_custom_plot_params(data_frame):
    numerical_cols = list(data_frame.columns)

    #grafiko rusis
    while True:
        print("\nChoose Chart Type:")
        print("  1) Bar Chart (X-axis fixed to Neighbourhood)")
        print("  2) Scatter Plot (X and Y axis user-defined)")
        chart_choice = input("Enter selection (1 or 2): ")

        if chart_choice == '1':
            chart_type = 'bar'
            x_column = 'hood'
            break
        elif chart_choice == '2':
            chart_type = 'scatter'
            break
        print("Invalid chart type. Please enter '1' or '2'.")

    #stulpeliu numeravimas vartotojo pasirinkimui
    def select_column_by_number(cols, axis_name):
        while True:
            print(f"\n--- Available columns for {axis_name} Axis ---")
            for i, col in enumerate(cols):
                print(f"  {i + 1}: {col}")

            try:
                selection = int(input(f"Choose {axis_name} column number (1-{len(cols)}): "))
                if 1 <= selection <= len(cols):
                    return cols[selection - 1]
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    #jei scatter plot duodamas pasirinkimas x asiai
    if chart_type == 'scatter':
        x_column = select_column_by_number(numerical_cols, "X")


    y_column = select_column_by_number(numerical_cols, "Y")

    #legendos pozicija
    legend_options = list(LEGEND_POSITIONS.keys())
    while True:
        print("\n    Legend Positions")
        for i, pos in enumerate(legend_options):
            print(f"  {i + 1}: {pos.title()}")

        try:
            pos_choice = int(input(f"Choose Legend Position number (1-{len(legend_options)}): "))
            if 1 <= pos_choice <= len(legend_options):
                legend_position = legend_options[pos_choice - 1]
                break
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return chart_type, x_column, y_column, legend_position


try:
    #duomenu nuskaitymas ir valymas
    df = pd.read_csv("listings.csv")
    df["price_clean"] = df["price"].apply(clean_price)
    df["price_pm_clean"] = df["price_per_month"].apply(clean_price)
    df["area_sqm"] = df["area_sqm"].apply(clean_area)
    df["rent_price"] = df["price_pm_clean"]
    df.loc[df["rent_price"].isna(), "purchase_price"] = df.loc[df["rent_price"].isna(), "price_clean"]
    df["purchase_price"] = df.get("purchase_price", pd.Series(index=df.index, dtype=float))
    df["purchase_price"] = df["purchase_price"].astype(float)

    #kv m kainos skaiciavimas
    df["rent_per_sqm"] = df["rent_price"] / df["area_sqm"]
    df["purchase_per_sqm"] = df["purchase_price"] / df["area_sqm"]

    #isimciu isemimas
    lower_r, upper_r = iqr_bounds(df["rent_per_sqm"])
    mask_r = df["rent_per_sqm"].between(lower_r, upper_r)
    lower_p, upper_p = iqr_bounds(df["purchase_per_sqm"])
    mask_p = df["purchase_per_sqm"].between(lower_p, upper_p)
    df["rent_per_sqm_clipped"] = df["rent_per_sqm"].where(mask_r, np.nan)
    df["purchase_per_sqm_clipped"] = df["purchase_per_sqm"].where(mask_p, np.nan)

    #agregavimas pagal rajona
    group = df.groupby("hood").agg(
        avg_rent_sqm=("rent_per_sqm_clipped", "mean"),
        avg_purchase_sqm=("purchase_per_sqm_clipped", "mean"),
        n_rent=("rent_per_sqm_clipped", lambda x: x.notna().sum()),
        n_purchase=("purchase_per_sqm_clipped", lambda x: x.notna().sum())
    )
    group = group[(group["avg_rent_sqm"].notna()) & (group["avg_purchase_sqm"].notna())]

    if group.empty:
        print("Error: No neighbourhood has both rent and purchase avg after cleaning.")
        sys.exit(1)

    #papildomi skaiciavimai 2 ir 3 grafikam
    group["rent_to_purchase_ratio"] = group["avg_rent_sqm"] / group["avg_purchase_sqm"]
    group["total_listings"] = group["n_rent"] + group["n_purchase"]

except FileNotFoundError:
    print("Error: The file 'listings.csv' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during data processing: {e}")
    sys.exit(1)

#vartotojo ivestis
while True:
    print("\n    Select a Chart to Generate:")
    print("1: Average Rent vs Purchase Scatter Plot")
    print("2: Top 10 Neighbourhoods by Rental Yield")
    print("3: Top 10 Neighbourhoods: Rental vs. Purchase Listing Counts")
    print("4: Custom Chart")
    print("0: Exit")
    selection = input("Enter your selection (1, 2, 3, 4, or 0): ")

    if selection == '1':
        plot_scatter(group)
    elif selection == '2':
        plot_yield(group)
    elif selection == '3':
        plot_counts(group)
    elif selection == '4':
        #vartotojo ivesties funkcija
        params = get_custom_plot_params(group)
        if params:
            chart_type, x_column, y_column, legend_position = params
            plot_custom(group, chart_type, x_column, y_column, legend_position)
    elif selection == '0':
        print("Exiting program.")
        break
    else:
        print("Invalid selection. Please enter 1, 2, 3, 4, or 0.")
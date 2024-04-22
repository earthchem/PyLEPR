# %%

import logging
import numpy as np
import pandas as pd

# %%


def begin_logging(log_filename):
    """
    Set up logging configuration.

    Parameters:
        log_filename (str): The name of the log file.
    """
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        format="--> %(levelname)s_(%(funcName)s): %(message)s",
        force=True,
    )


def print_log_file(log_filename):
    """
    Print the contents of the log file.

    Parameters:
        log_filename (str): The name of the log file.
    """
    try:
        with open(log_filename, "r") as fin:
            print(fin.read())
    except FileNotFoundError:
        print(f"Error: Log file '{log_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def idx_to_col(idx):
    """
    Converts zero-indexed column number to Excel column label.
    """
    if idx < 26:
        return chr(65 + idx)
    else:
        return idx_to_col((idx // 26) - 1) + idx_to_col(idx % 26)


def col_to_idx(col):
    """
    Converts Excel column label to a zero-indexed column number.
    """
    index = 0
    for i, char in enumerate(reversed(col.upper())):
        index += (ord(char) - 65 + 1) * (26**i)
    return index - 1


def extract_chem_dat(upload_data, sheet, start_col, start_row):
    """
    Extracts and organizes chemical data from a specified sheet in the provided dataset.

    Parameters:
        upload_data (dict): Contains DataFrames indexed by sheet names.
        sheet (str): The name of the sheet to extract data from.
        start_col (str): The column letter where chemical data starts.
        start_row (int): The row index where chemical data headers are located.

    Returns:
        tuple: A tuple containing two DataFrames, (chem_dat, chem_dat_info):
               - chem_dat: DataFrame with chemical data.
               - chem_dat_info: DataFrame with metadata such as method IDs and units.
    """

    run_products = upload_data[sheet]

    start_col_idx = col_to_idx(start_col)
    run_names = run_products.iloc[start_row:, 0]

    dat = run_products.iloc[:, start_col_idx:]
    dat.columns = dat.iloc[0]
    dat = dat.iloc[1:]

    chem_dat_info = dat.iloc[:2]
    chem_dat_info.index = ["method_id", "unit"]

    chem_dat = dat.iloc[start_row - 1 :]
    chem_dat
    chem_dat.index = run_names

    return chem_dat, chem_dat_info


# %%


def validate_column_names(chem_dat_info, start_col="H"):
    """
    Validates that column names in a DataFrame adhere to preferred naming conventions, ensures that totals are volatile-free.
    """

    preferred_formats = {
        "FeO": ["FeOt", "FeOtot", "FeOtotal", "FeO*"],
        "Fe2O3": ["Fe2O3t", "Fe2O3tot", "Fe2O3total", "Fe2O3*"],
    }

    for column in chem_dat_info.columns:
        correct_name = None
        for preferred, variations in preferred_formats.items():
            if column in variations:
                correct_name = preferred
                break

        if correct_name:
            excel_column = idx_to_col(
                chem_dat_info.columns.get_loc(column) + col_to_idx(start_col)
            )
            logging.error(
                f"Column '{column}' (Excel Column {excel_column}) is not formatted correctly. Please replace with '{correct_name}'."
            )

        if column == "Total" or column == "total":
            excel_column = idx_to_col(
                chem_dat_info.columns.get_loc(column) + col_to_idx(start_col)
            )
            logging.warning(
                f"Column '{column}' (Excel Column {excel_column}) detected. Please ensure this is the volatile-free total."
            )


def validate_chem_error_columns(chem_dat_info):
    """
    Validates that each chemical data column has a corresponding error column, ending in '_err' or 'error'.
    """
    columns = chem_dat_info.columns
    # Find columns that don't already end with '_err' or 'error'
    meas_cols = [
        col for col in columns if not (col.endswith("_err") or col.endswith("error"))
    ]

    for col in meas_cols:
        # Check for both possible error column names
        error_col_suffixes = ["_err", "error"]
        found_error_col = any(
            (col + suffix in columns) for suffix in error_col_suffixes
        )

        if not found_error_col:
            # If neither error column is found, log a warning for the first expected name
            logging.warning(
                f"'{col}_err' or '{col}error' missing from chemistry data columns."
            )


def validate_chem_units(chem_dat_info, start_col="H"):
    """
    Validates that each chemical data column has units specified.
    """
    units = chem_dat_info.loc["unit"]
    for idx, (col_name, unit) in enumerate(units.items(), start=col_to_idx(start_col)):
        if pd.isna(unit):
            excel_column = idx_to_col(idx)  # Calculate the Excel column letter
            logging.critical(
                f"'{col_name}' (Excel Column {excel_column}) does not provide any units."
            )


def validate_chem_method(chem_dat_info, start_col="H"):
    """
    Validates that each chemical data column has units specified.
    """
    methods = chem_dat_info.loc["method_id"]
    for idx, (col_name, method) in enumerate(
        methods.items(), start=col_to_idx(start_col)
    ):
        if pd.isna(method):
            excel_column = idx_to_col(idx)  # Calculate the Excel column letter
            logging.error(
                f"'{col_name}' (Excel Column {excel_column}) does not provide any method_id."
            )


def validate_metadata(chem_dat_info):
    """
    Executes all validation functions on the chem_dat_info.
    """
    try:
        validate_column_names(chem_dat_info)
        validate_chem_error_columns(chem_dat_info)
        validate_chem_units(chem_dat_info)
        validate_chem_method(chem_dat_info)
        print("Metadata validation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during validation: {e}")
        print(f"An error occurred during validation: {e}")


# %%


def validate_chemical_values(chem_dat, start_col="H", start_row=7):
    """
    Validates chemical data values within a DataFrame, logging any formatting issues or non-numeric errors.
    The function will print and log a completion message after processing all entries.

    Parameters:
        chem_dat (DataFrame): DataFrame containing the chemical data.
        start_col (str): Column letter where chemical data starts (default 'H').
        start_row (int): Row index where data entries begin (default 7).
    """
    try:
        start_col_idx = col_to_idx(start_col)
        for ichem_col, ichem_dat in chem_dat.T.iterrows():
            chem = ichem_dat.name
            column_index = chem_dat.columns.get_loc(chem) + start_col_idx
            excel_column = idx_to_col(column_index)

            for run_id, val in ichem_dat.items():
                row_number = chem_dat.index.get_loc(run_id) + start_row
                cell_location = f"{excel_column}{row_number}"
                message = validate_value(val, cell_location)

                if message:
                    logging.error(message)
                    continue  # Skip further checks if an error is already found
        
        print("Chemical data validation completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during chemical data validation: {e}")
        print(f"An error occurred during chemical data validation: {e}")


def validate_value(val, cell_location):
    """
    Validates a single chemical value, returning an appropriate error message if issues are found.

    Parameters:
        val: The value to validate.
        cell_location (str): The Excel cell location of the value for logging purposes.

    Returns:
        str: An error message if the value is invalid, None otherwise.
    """

    if isinstance(val, str):
        val_lower = val.lower()
        # Checking for symbols indicating limits and their correctness
        if val_lower.startswith((">", "<", "≤", "≥")):
            # Log initial detection of symbol
            initial_message = f"'{val}' (Excel Cell {cell_location}) contains a symbol."
            logging.warning(initial_message)

            # Check numeric validity of the part after the symbol
            numeric_part = val[1:].strip()
            if not numeric_part.replace('.', '', 1).isdigit():
                return f"'{val}' (Excel Cell {cell_location}) is not valid, as it is not numeric."
            else:
                return None  # If valid, no further checks are needed

    if isinstance(val, str):  # Check if the value is a string to handle string-specific validations
        val_lower = val.lower()  # Convert value to lowercase to standardize checks
        # Check for non-detects and placeholder values
        if val_lower in ["nd", "n.d.", "n.d"]:
            return f"'{val}' (Excel Cell {cell_location}) is not valid. Use 'bdl' if below detection limit."
        # Check for and validate symbols indicating limits
        elif val_lower.startswith("≤"):
            return f"'{val}' (Excel Cell {cell_location}) is not valid. Replace '≤' with '<='."
        elif val_lower.startswith("≥"):
            return f"'{val}' (Excel Cell {cell_location}) is not valid. Replace '≥' with '>='."
        elif any(val_lower.startswith(sym) for sym in [">", "<", ">=", "<="]):
            # Validate the numeric part after the symbol
            numeric_part = val[1:].strip()
            if not numeric_part.replace('.', '', 1).isdigit():
                return f"'{val}' (Excel Cell {cell_location}) is not valid. The part after '{val[0]}' must be numeric."
        elif "≌" in val_lower:
            return f"'{val}' (Excel Cell {cell_location}) is not valid. Remove ≌."
    # Handling for placeholders indicating no measurement
    elif val == "-":
        return f"'-' in (Excel Cell {cell_location}) is not valid. Leave cell blank if not measured."
    # Handling zero with specific instructions
    elif val == 0:
        return f"0 in (Excel Cell {cell_location}) is not valid. Use 'bdl' for below detection limit values and leave cell blank if not measured."
    return None


# %%
# %%

import re
import logging
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# %%


def begin_logging(log_filename):
    """
    Set up logging configuration to capture all logs with distinct formatting for INFO logs.
    Each log level will have a specific formatter to distinguish INFO logs by formatting them without arrows.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to capture all levels of logs

    # Remove all existing handlers associated with the logger.
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    # Create a single file handler for all log messages
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)  # This handler captures all levels of logs

    # Define a method to differentiate formatting based on log level
    def format(record):
        if record.levelno == logging.INFO:
            return f"{record.getMessage()}"
        else:
            return f"---> {record.levelname} ({record.funcName}): {record.getMessage()}"

    # Set a custom formatter using the format function
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.addFilter(lambda record: (format(record), setattr(record, 'msg', format(record)))[0])

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


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
    Correctly handles indices greater than 25 to account for Excel's labeling.
    """
    if idx < 26:
        return chr(65 + idx)
    else:
        return idx_to_col(idx // 26 - 1) + chr(65 + (idx % 26))


def col_to_idx(col):
    """
    Converts Excel column label to a zero-indexed column number.
    """
    index = 0
    for i, char in enumerate(reversed(col.upper())):
        index += (ord(char) - 65 + 1) * (26**i)
    return index - 1


def extract_chem_data(upload_data, sheet, start_col, start_row):
    """
    Extracts and organizes chemical data from a specified sheet in the provided dataset.

    Parameters:
        upload_data (dict): Contains DataFrames indexed by sheet names.
        sheet (str): The name of the sheet to extract data from.
        start_col (str): The column letter where chemical data starts.
        start_row (int): The row index where chemical data headers are located.

    Returns:
        tuple: A tuple containing two DataFrames, (chem_data, chem_data_info):
               - chem_data: DataFrame with chemical data.
               - chem_data_info: DataFrame with metadata such as method IDs and units.
    """

    run_products = upload_data[sheet]
    column_names = run_products.iloc[start_row - 4]
    run_products.columns = column_names.values

    start_col_idx = col_to_idx(start_col)
    run_names = run_products.iloc[start_row - 2 :, 0 : col_to_idx(start_col)]

    data = run_products.iloc[:, start_col_idx:]
    data.columns = data.iloc[0]
    data = data.iloc[1:]

    chem_data_info = data.iloc[:2]
    chem_data_info.index = ["method_id", "unit"]

    chem_data = data.iloc[start_row - 3 :]
    chem_data.index = run_names.iloc[:, 0]

    for i in range(
        1, run_names.shape[1]
    ):  # Loop through columns, starting from the second column
        chem_data.insert(i - 1, run_names.columns[i], run_names.iloc[:, i].values)

    return chem_data, chem_data_info


def extract_data(upload_data, sheet, start_col, start_row):
    """
    Extracts and organizes data from a specified sheet in the provided dataset.

    Parameters:
        upload_data (dict): Contains DataFrames indexed by sheet names.
        sheet (str): The name of the sheet to extract data from.
        start_col (str): The column letter where data starts.
        start_row (int): The row index where data headers are located.

    Returns:
        dat: Dataframe from the sheet with the first column as the index.
    """

    start_col_idx = col_to_idx(start_col)
    run_products = upload_data[sheet]
    column_names = run_products.iloc[start_row - 5, start_col_idx:]
    run_products.columns = column_names.values

    data = run_products.iloc[start_row - 2 :, :]
    data = data.set_index(data.columns[0])

    return data


# %%


def phase_scrape(link):
    """
    Extracts data from the phase table within an HTML file and converts it to a DataFrame.

    Parameters:
        link (str): The file path or link to the HTML file to be scraped.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted table data, where each row represents
                          a data record from the table and each column represents a field in the table.
    """

    # Open the HTML file
    with open(link) as fp:
        # Use a parser (html.parser) with BeautifulSoup
        soup = BeautifulSoup(fp, "html.parser")

    # Find the table by its role attribute
    table = soup.find("table", {"role": "table"})

    # Initialize a list to hold all rows of data
    table_data = []

    # Extract headers
    headers = [th.text.strip() for th in table.find_all("th")]

    # Extract all rows in the table
    rows = table.find_all("tr")

    # Iterate over each row
    for row in rows:
        # Extract text from each cell in the row
        cols = [ele.text.strip() for ele in row.find_all("td")]
        # Append non-empty lists to the table_data
        if cols:
            table_data.append(cols)

    if not headers or not table_data:
        raise ValueError("Table is missing headers or rows.")

    df = pd.DataFrame(table_data, columns=headers)

    return df


# %%


def validate_column_names(chem_data_info, start_col):
    """
    Validates that column names in a DataFrame adhere to preferred naming conventions, ensures that totals are volatile-free.
    """

    preferred_formats = {
        "FeO": ["FeOt", "FeOtot", "FeOtotal", "FeO*"],
        "Fe2O3": ["Fe2O3t", "Fe2O3tot", "Fe2O3total", "Fe2O3*"],
    }

    for column in chem_data_info.columns:
        correct_name = None
        for preferred, variations in preferred_formats.items():
            if column in variations:
                correct_name = preferred
                break

        if correct_name:
            excel_column = idx_to_col(
                chem_data_info.columns.get_loc(column) + col_to_idx(start_col)
            )
            logging.error(
                f"Column '{column}' (Excel Column {excel_column}) is not formatted correctly. Please replace with '{correct_name}'."
            )

        if column == "Total" or column == "total":
            excel_column = idx_to_col(
                chem_data_info.columns.get_loc(column) + col_to_idx(start_col)
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


def validate_chem_units(chem_data_info, start_col):
    """
    Validates that each chemical data column has units specified.
    """
    units = chem_data_info.loc["unit"]
    for idx, (col_name, unit) in enumerate(units.items(), start=col_to_idx(start_col)):
        if pd.isna(unit):
            excel_column = idx_to_col(idx)  # Calculate the Excel column letter
            logging.critical(
                f"'{col_name}' (Excel Column {excel_column}) does not provide any units."
            )


def validate_chem_method(chem_data_info, start_col):
    """
    Validates that each chemical data column has units specified.
    """
    methods = chem_data_info.loc["method_id"]
    for idx, (col_name, method) in enumerate(
        methods.items(), start=col_to_idx(start_col)
    ):
        if pd.isna(method):
            excel_column = idx_to_col(idx)  # Calculate the Excel column letter
            logging.error(
                f"'{col_name}' (Excel Column {excel_column}) does not provide any method_id."
            )


def validate_units(chem_data_info, start_col):
    """
    Validates that the units in a DataFrame match an accepted list of units.
    Logs a warning if any unit values do not match and skips NaN values.

    Parameters:
        data (DataFrame): DataFrame containing the unit data.
        start_col (str): Excel column where data headers start.
    """

    # Scrape the accepted units
    df = phase_scrape(link="../data/units.html")
    accepted_units = df["name"].str.lower().tolist()

    # Ensure there is a row labeled 'unit'
    if 'unit' in chem_data_info.index:
        unit_values = chem_data_info.loc['unit'].values  # Extract unit values from the row
        for i, unit in enumerate(unit_values):
            if pd.isna(unit):  # Skip NaN values
                continue
            unit_str = str(unit).lower()  # Convert to lowercase for case-insensitive comparison
            if unit_str not in accepted_units:
                excel_column = idx_to_col(i + col_to_idx(start_col))  # Convert column index to Excel label with offset
                logging.warning(
                    f"'{unit}' (Excel Column {excel_column}) does not match any accepted units. Please check for typos or reexamine the list on the GitHub Wiki."
                )


def validate_metadata(chem_data_info, start_col):
    """
    Executes all validation functions on the chem_dat_info.
    """
    try:
        validate_column_names(chem_data_info, start_col)
        validate_chem_error_columns(chem_data_info)
        validate_chem_units(chem_data_info, start_col)
        validate_chem_method(chem_data_info, start_col)
        validate_units(chem_data_info, start_col)
        # print("Metadata validation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during validation: {e}")
        print(f"An error occurred during validation: {e}")


# %%


def validate_chem_phase_values(chem_data, start_row):
    """
    Validates chemical data values within a DataFrame, logging any formatting issues or non-numeric errors.
    The function will print and log a completion message after processing all entries.

    Parameters:
        chem_data (DataFrame): DataFrame containing the chemical data.
        start_col (str): Column letter where chemical data starts (default).
        start_row (int): Row index where data entries begin (default).
    """
    try:
        validate_required_fields(chem_data, start_row)

        for ichem_col, ichem_data in chem_data.T.iterrows():
            chem = ichem_data.name
            excel_column = idx_to_col(chem_data.columns.get_loc(chem) + 1)

            for run_id, val in ichem_data.items():
                row_number = chem_data.index.get_loc(run_id) + start_row
                cell_location = f"{excel_column}{row_number}"

                if chem == "SPECIES":
                    message = validate_full_phase(val, cell_location)
                elif chem == "Phase List":
                    message = validate_phase_list(val, cell_location)
                else:
                    message = validate_value(val, cell_location)

                if message:
                    logging.error(message)
                    continue  # Skip further checks if an error is already found

        # print("Chemical data validation completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during chemical data validation: {e}")
        print(f"An error occurred during chemical data validation: {e}")


def validate_symbol(val, cell_location):
    """Validates the symbol and numeric part of the given value."""
    symbol_mapping = {'≤': '<=', '≥': '>='}
    if val[0] in symbol_mapping:
        return f"'{val}' (Excel Cell {cell_location}) is not valid. Replace '{val[0]}' with '{symbol_mapping[val[0]]}'."

    if val[0] in "<>":
        numeric_part = val[1:].strip()
        if not numeric_part.replace(".", "", 1).isdigit():
            return f"'{val}' (Excel Cell {cell_location}) is not valid. The part after '{val[0]}' must be numeric."


def validate_value(val, cell_location):
    if isinstance(val, str):
        val_lower = val.lower()

        if "(" in val_lower or ")" in val_lower:
            return f"'{val}' (Excel Cell {cell_location}) is not valid. Please provide an absolute uncertainty."

        if val_lower in ["nd", "n.d.", "n.d"]:
            return f"'{val}' (Excel Cell {cell_location}) is not valid. Use 'bdl' if below detection limit."

        if val_lower == "-":
            return f"'-' in (Excel Cell {cell_location}) is not valid. Leave cell blank if not measured."

        if val_lower == "0":
            return f"0 in (Excel Cell {cell_location}) is not valid. Use 'bdl' for below detection limit values and leave cell blank if not measured."

        if "≌" in val_lower:
            return f"'{val}' (Excel Cell {cell_location}) is not valid. Remove ≌."

        # Symbol-based validation
        symbol_error = validate_symbol(val, cell_location)
        if symbol_error:
            return symbol_error



def validate_full_phase(val, cell_location):
    """
    Validates a chemical phase value against a list of accepted phases, returning an error message if the value does not match any accepted phase.

    Parameters:
        val: The value to validate.
        cell_location (str): The Excel cell location of the value for logging purposes.

    Returns:
        str: An error message if the value is invalid, None otherwise.
    """

    df = phase_scrape(link="../data/pylepr_phases.html")
    accepted_phases = df["Description"].str.lower().tolist()

    if isinstance(val, str):
        val_lower = val.lower()  # Convert value to lowercase to standardize comparison
        if val_lower not in accepted_phases:
            return f"'{val}' (Excel Cell {cell_location}) contains invalid full phase names. Please check for typos or reexamine the list on the GitHub Wiki."
    return None


def validate_phase_list(val, cell_location):
    """
    Validates a list of chemical phase abbreviations against a list of accepted phases,
    returning an error message if any part of the list does not match the accepted phases.

    Parameters:
        val: The string containing the phase abbreviations, separated by commas or plus signs.
        cell_location (str): The Excel cell location of the value for logging purposes.

    Returns:
        str: An error message if any abbreviation is invalid, None otherwise.
    """

    # Fetch the list of accepted phases
    df = phase_scrape(link="../data/pylepr_phases.html")
    accepted_phases = df["Abbreviation"].str.lower().tolist()

    # Standardize the value to lowercase and split by ',' or '+'
    if isinstance(val, str):
        # Split the value using a regex to handle multiple delimiters
        phases = re.split(r"[+,;]", val.lower())
        phases = [phase.strip() for phase in phases]  # Strip whitespace from each part

        # Check each part against the list of accepted phases
        invalid_phases = [phase for phase in phases if phase not in accepted_phases]
        if invalid_phases:
            invalid_list = ", ".join(invalid_phases)
            return f"'{invalid_list}' (Excel Cell {cell_location}) contains invalid phase abbreviations. Please check for typos or reexamine the list on the GitHub Wiki."

    return None


# %%


# %%


def validate_required_fields(chem_data, start_row):
    """
    Validates that required fields (in uppercase) in a DataFrame are fully populated.
    Logs a warning if any required fields are missing, including specific cell locations.

    Parameters:
        chem_data (DataFrame): DataFrame containing the chemical data.
        start_row (int): Row index where data headers are located (default to the top of the data section).
    """

    required_columns = [col for col in chem_data.columns if col.isupper()]

    for col in required_columns:
        if chem_data[col].isnull().any():
            for i, is_null in enumerate(chem_data[col].isnull()):
                if is_null:
                    row_number = (
                        i + start_row
                    )  # Calculating actual row number in the sheet
                    excel_column = idx_to_col(
                        chem_data.columns.get_loc(col) + 1
                    )  # Get Excel column
                    cell_location = f"{excel_column}{row_number}"
                    logging.error(
                        f"Missing value found at Excel Cell {cell_location}. Please provide a value for '{col}'."
                    )


def validate_buffer(data, start_row):
    """
    Validates that the fO2 values in a DataFrame match an accepted list of abbreviations or is numeric.
    Logs a warning if any fO2 values do not match.

    Parameters:
        data (DataFrame): DataFrame containing the buffer data.
        start_row (int): Row index where data headers are located.
    """

    # Scrape the accepted buffer values
    df = phase_scrape(link="../data/fO2.html")
    accepted_buffer = df["Abbreviation"].str.lower().tolist()

    # Check each buffer value in the DataFrame
    if 'fO2' in data.columns:
        for i, buffer_value in enumerate(data['fO2']):
            buffer_value_str = str(buffer_value).lower()  # Ensure numeric values are converted to strings for processing
            try:
                float(buffer_value)  # Try to convert to float
                continue  # If successful, it's numeric and valid, continue to next iteration
            except ValueError:
                # If conversion fails, check if it matches any abbreviation
                if not any(abbreviation in buffer_value_str for abbreviation in accepted_buffer):
                    row_number = i + start_row  # Adjust row number based on starting row index
                    excel_column = idx_to_col(data.columns.get_loc('fO2') + 1)  # Excel column index
                    cell_location = f"{excel_column}{row_number}"
                    logging.warning(
                    f"'{buffer_value}' (Excel Cell {cell_location}) does not match any accepted buffers. Please check for typos or reexamine the list on the GitHub Wiki."
                    )


def validate_methods(data, start_row):
    """
    Validates that the TECHNIQUE values in a DataFrame match an accepted list of techniques.
    Logs a warning if any TECHNIQUE values do not match.

    Parameters:
        data (DataFrame): DataFrame containing the technique data.
        start_row (int): Row index where data headers are located.
    """

    # Scrape the accepted techniques
    df = phase_scrape(link="../data/methods.html")
    accepted_methods = df["name"].str.lower().tolist()

    if 'TECHNIQUE' in data.columns:
        for i, technique in enumerate(data['TECHNIQUE']):
            technique_str = str(technique).lower()  # Convert to lowercase for case-insensitive comparison
            if technique_str not in accepted_methods:
                row_number = i + start_row  # Adjust row number based on starting row index
                excel_column = idx_to_col(data.columns.get_loc('TECHNIQUE') + 1)  # Excel column index
                cell_location = f"{excel_column}{row_number}"
                logging.warning(
                    f"'{technique}' (Excel Cell {cell_location}) does not match any accepted techniques. Please check for typos or reexamine the list on the GitHub Wiki."
                )


# %%


#def validate_device_codes(data_device, primary_device, secondary_device):
def validate_device_codes(data_device, data_metadata):
    """
    Validates the consistency of 'DEVICE' codes between two data sources. It logs errors for any device codes
    that are unique to either the device data or metadata, indicating mismatches between the two sets.

    Parameters:
        data_device (DataFrame): DataFrame containing the device data with a 'DEVICE' column.
        data_metadata (DataFrame): DataFrame containing the metadata with device codes in the index.
    """

    unique_devices = np.unique(data_device["DEVICE"].astype(str))
    unique_devices_metadata = np.unique(data_metadata.index.astype(str))

    # Find unique in unique_devices not in unique_devices_metadata
    unique_to_devices = np.setdiff1d(unique_devices, unique_devices_metadata)
    if unique_to_devices.size > 0:
        logging.error(
            f"Unique DEVICE codes {unique_to_devices} in Sheet '2 Experiments' not found in Sheet '5 Device Metadata'."
        )

    # Find unique in unique_devices_metadata not in unique_devices
    unique_to_metadata = np.setdiff1d(unique_devices_metadata, unique_devices)
    if unique_to_metadata.size > 0:
        logging.error(
            f"Unique DEVICE codes {unique_to_metadata} in Sheet '5 Device Metadata' not found in Sheet '2 Experiments'."
        )


def validate_method_codes(
    starting_materials, run_products, data, primary_methods, secondary_methods
):
    unique_data_methods = np.unique(
        np.concatenate(
            [
                starting_materials.loc["method_id"].values.astype(str),
                run_products.loc["method_id"].values.astype(str),
                data.loc["method_id"].values.astype(str),
            ]
        )
    )

    unique_methods = np.unique(
        np.concatenate(
            [primary_methods.index.astype(str), secondary_methods.index.astype(str)]
        )
    )

    unique_to_data_methods = np.setdiff1d(unique_data_methods, unique_methods)
    if unique_to_data_methods.size > 0:
        logging.error(
            f"Unique METHOD CODE {unique_to_data_methods} in Sheets '3 Bulk (Starting Materials), 4 Bulk (Run Products), or 7 Data' not found in Sheets '8 Primary Method Metadata or 9 Method-specific Metadata'."
        )

    unique_to_methods = np.setdiff1d(unique_methods, unique_data_methods)
    if unique_to_methods.size > 0:
        logging.error(
            f"Unique METHOD CODE {unique_to_methods} in Sheets '8 Primary Method Metadata or 9 Method-specific Metadata' not found in Sheets '3 Bulk (Starting Materials), 4 Bulk (Run Products), or 7 Data'."
        )


# %%


def validate_all(upload_data, log_filename):

    """
    Validates various data sheets within an uploaded data workbook and logs the validation process.

    Parameters:
        upload_data (object): The workbook object containing the data to be validated.
        log_filename (str): The path and name of the log file where the validation process details will be recorded.

    Description:
        This function initializes the logging configuration, extracts and validates data from specific sheets within the provided workbook.
        It logs the start and completion of validation for each sheet, ensuring that all necessary validations are performed and recorded sequentially for '2 Experiments', '3 Bulk (Starting Materials)', '4 Bulk (Run Products)', '5 Device Metadata', '6 Data', '7 Primary Method Metadata', and '8 Method-Specific Metadata'.
    """

    begin_logging(log_filename)

    dat_2 = extract_data(upload_data, sheet='2 Experiments', start_col='A', start_row=7)
    chem_data_3, chem_data_info_3 = extract_chem_data(upload_data, sheet='3 Bulk (Starting Materials)', start_col='H', start_row=7)
    chem_data_4, chem_data_info_4 = extract_chem_data(upload_data, sheet='4 Bulk (Run Products)', start_col='F', start_row=7)
    dat_5 = extract_data(upload_data, sheet='5 Primary Device Metadata', start_col='A', start_row=7)
    dat_6 = extract_data(upload_data, sheet='6 Device-specific Metadata', start_col='A', start_row=7)
    chem_data_7, chem_data_info_7 = extract_chem_data(upload_data, sheet='7 Data', start_col='H', start_row=7)
    dat_8 = extract_data(upload_data, sheet='8 Primary Method Metadata', start_col='A', start_row=7)
    dat_9 = extract_data(upload_data, sheet='9 Method-specific Metadata', start_col='A', start_row=7)

    logging.info("\nSTARTING VALIDATION FOR SHEET '2 Experiments'\n")
    validate_required_fields(dat_2, start_row=7)
    validate_buffer(dat_2, start_row=7)

    logging.info("\nSTARTING VALIDATION FOR SHEET '3 Bulk (Starting Materials)'\n")
    validate_metadata(chem_data_info_3, start_col='H')
    validate_chem_phase_values(chem_data_3, start_row=7)

    logging.info("\nSTARTING VALIDATION FOR SHEET '4 Bulk (Run Products)'\n")
    validate_metadata(chem_data_info_4, start_col='F')
    validate_chem_phase_values(chem_data_4, start_row=7)

    #logging.info("\nSTARTING VALIDATION FOR SHEET '5 Primary Device Metadata' and '6 Device-specific Metadata'\n")
    #validate_device_codes(dat_2, dat_5, dat_6)

    logging.info("\nSTARTING VALIDATION FOR SHEET '7 Data'\n")
    validate_metadata(chem_data_info_7, start_col='H')
    validate_chem_phase_values(chem_data_7, start_row=7)

    logging.info("\nSTARTING VALIDATION FOR SHEET '8 Primary Method Metadata' and '9 Method-specific Metadata'\n")
    validate_method_codes(chem_data_info_3, chem_data_info_4, chem_data_info_7, dat_8, dat_9)
    validate_methods(dat_8, start_row=7)

    logging.info("\nVALIDATION COMPLETE FOR ALL SHEETS\n")


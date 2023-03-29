import logging

import numpy as np
import pandas as pd

log_filename = 'validation.log'
logging.basicConfig(filename=log_filename,
                    filemode='w',
                    format="---> %(levelname)s_(%(funcName)s):: %(message)s.")

def print_log_file(log_filename=log_filename):
    with open(log_filename, 'r') as fin:
        print(fin.read())
        
        
def extract_chem_dat(upload_data):
    run_products = upload_data['6 Run Products']
    
    header_row_num = 4
    chem_dat_col_index = 13
    run_names = run_products.iloc[header_row_num+1:,0]

    dat = run_products.iloc[:,chem_dat_col_index:]
    dat.columns = dat.iloc[0]
    dat = dat.iloc[1:]
    chem_dat_info = dat.iloc[:2]
    chem_dat_info.index = ['method_id','unit']

    chem_dat = dat.iloc[header_row_num:]
    chem_dat
    chem_dat.index = run_names

    return chem_dat, chem_dat_info


def _validate_chem_error_columns(chem_dat_info):
    columns = chem_dat_info.columns
    meas_cols = [col for col in columns if not col.endswith('_err') ]
    for col in meas_cols:
        if col+'_err' not in columns:
            logging.error(f"'{col}_err' missing from chemistry data columns")
            

def _validate_chem_units(chem_dat_info):
    for (col, dat) in chem_dat_info.T.iterrows():
        if dat.unit is np.nan:
            logging.critical(f"'{col}' does not provide any units")
            
def _validate_chem_method(chem_dat_info):
    for (col, dat) in chem_dat_info.T.iterrows():
        if dat.method_id is np.nan:
            logging.critical(f"'{col}' does not provide any method id")
            

def validate_chem_data_info(chem_dat_info):
    _validate_chem_error_columns(chem_dat_info)
    _validate_chem_units(chem_dat_info)
    _validate_chem_method(chem_dat_info)
    
    
    
    
# ichem_dat = chem_dat.iloc[:,1]

# def validate_numeric_chem_data(ichem_dat):

        
 
def _chem_not_detected_not_valid(val, chem, run_id):
    if val=='nd':
        logging.error(f"'{val}', the '{chem}' value for exp_run '{run_id}', is not valid. If not detected use vocabulary 'bdl'")
        return True
    
    return False

def _chem_not_measured_not_valid(val, chem, run_id):
    if val=='-':
        logging.error(f"'{val}', the '{chem}' value for exp_run '{run_id}', is not valid. If not measured leave entry blank")
        return True
    
    return False

def _chem_measurement_limit_not_valid(val, chem, run_id):
    if type(val) is not str:
        return False
    
    if val.startswith('>') or val.startswith('<'):
        logging.error(f"'{val}', the '{chem}' value for exp_run '{run_id}', is not valid. Instead give just the value and indicate limit using field '????, Ask roger'")
        return True
    
    return False
        
def _numeric_chem_data_not_valid(val, chem, run_id):
    if type(val) is str:
        logging.error(f"'{val}', the '{chem}' value for exp_run '{run_id}', is not a valid number")
        return True
    
    return False


def validate_chem_data(chem_dat):
    for ichem_col, ichem_dat in chem_dat.T.iterrows():
        chem = ichem_dat.name
        for run_id, val in ichem_dat.items():
            if _chem_not_detected_not_valid(val, chem, run_id):
                continue
            
            if _chem_not_measured_not_valid(val, chem, run_id):
                continue
            
            if _chem_measurement_limit_not_valid(val, chem, run_id):
                continue
            
            _numeric_chem_data_not_valid(val, chem, run_id)
            
            
def validate_upload(upload_data):
    chem_dat, chem_dat_info = extract_chem_dat(upload_data)
    validate_chem_data_info(chem_dat_info)
    validate_chem_data(chem_dat)
    print_log_file()
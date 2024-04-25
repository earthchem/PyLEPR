# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Validation Notebook for LEPR Spreadsheet Uploads

# %%
import sys
import pandas as pd

sys.path.append('../src/')
import pylepr.validate as pl

# %%
upload_data = pd.read_excel(
    "../data/upload_validation.xlsx",
    sheet_name=None,
    engine="openpyxl",
)

# %%
log_filename = 'validation_all.log'
pl.validate_all(upload_data, log_filename)
pl.print_log_file(log_filename)

# %%

# %%

# %%

# %%

# %%

# %%
chem_data_6, chem_data_info_6 = pl.extract_chem_data(upload_data, sheet='6 Data', start_col='H', start_row=7)
display(chem_data_6)
display(chem_data_info_6)

log_filename_6 = "validation_sheet6.log"
pl.begin_logging(log_filename_6)

pl.validate_metadata(chem_data_info_6, start_col='H')

pl.validate_chem_phase_values(chem_data_6, start_row=7)

pl.print_log_file(log_filename_6)

# %%
chem_data_4, chem_data_info_4 = pl.extract_chem_data(upload_data, sheet='4 Bulk (Run Products)', start_col='F', start_row=7)
display(chem_data_4)
display(chem_data_info_4)

log_filename_4 = "validation_sheet4.log"
pl.begin_logging(log_filename_4)

pl.validate_metadata(chem_data_info_4, start_col='F')

pl.validate_chem_phase_values(chem_data_4, start_row=7)

pl.print_log_file(log_filename_4)

# %%
dat_2 = pl.extract_data(upload_data, sheet='2 Experiments', start_col='A', start_row=7)
display(dat_2)

log_filename_2 = "validation_sheet2.log"
pl.begin_logging(log_filename_2)

pl.validate_required_fields(dat_2, start_row=7)

pl.validate_buffer(dat_2, start_row=7)

pl.print_log_file(log_filename_2)

# %%
chem_data_3, chem_data_info_3 = pl.extract_chem_data(upload_data, sheet='3 Bulk (Starting Materials)', start_col='H', start_row=7)
display(chem_data_3)
display(chem_data_info_3)

log_filename_3 = "validation_sheet3.log"
pl.begin_logging(log_filename_3)

pl.validate_metadata(chem_data_info_3, start_col='H')

pl.validate_chem_phase_values(chem_data_3, start_row=7)

pl.print_log_file(log_filename_3)

# %%
dat_5 = pl.extract_data(upload_data, sheet='5 Device Metadata', start_col='A', start_row=7)
display(dat_5)

log_filename_5 = "validation_sheet5.log"

pl.begin_logging(log_filename_5)

pl.validate_device_codes(dat_2, dat_5)

pl.print_log_file(log_filename_5)

# %%
dat_7 = pl.extract_data(upload_data, sheet='7 Primary Method Metadata', start_col='A', start_row=7)
dat_8 = pl.extract_data(upload_data, sheet='8 Method-Specific Metadata', start_col='A', start_row=7)

log_filename_78 = "validation_sheet78.log"

pl.begin_logging(log_filename_78)

pl.validate_method_codes(chem_data_info_3, chem_data_info_4, chem_data_info_6, dat_7, dat_8)

pl.print_log_file(log_filename_78)


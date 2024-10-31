"""

This file allows conversion of file names as downloaded from Elia / ENTSOE data platforms to be uniformized to historical file names


"""


import os

def rename_files_in_folder(folder_path, target_rules):
    #Function looping through all files in a folder, triggering rules for renaming the files based on a dictionary
    for filename in os.listdir(folder_path):
        for target_string, rename_rule in target_rules.items():
            if target_string in filename:
                target_filename = rename_rule(filename)
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, target_filename)

                if filename != target_filename: #Only
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed {old_file_path} to {new_file_path}")
                    break  # Stop looking for other substrings once a match is found

# Rules for target filenames
def rule_gen_prod(filename):
    #Rule for Elia filenames of historical production
    year = filename[-11:-7]
    month = filename[-7:-4]
    return f"Generation_Produced_Historical_{year}-{month}.XLS"

def rule_gen_fc(filename):
    #Rule for Elia filenames of historical production forecast
    dict_conversion = {
        "January": "01",
        "February": "02",
        "March": "03",
        "April": "04",
        "May": "05",
        "June": "06",
        "July": "07",
        "August": "08",
        "September": "09",
        "October": "10",
        "November": "11",
        "December": "12",
    }

    for month in list(dict_conversion.keys()):
        if month in filename:
            m_nb = dict_conversion[month]
            y_nb = filename[-8:-4]
            return f"Generation_Forecast_Historical_Cipu_{m_nb}_{y_nb}.XLS"

    return filename.replace(".txt", "_rule2.txt")

def rule_imb_nrv(filename):
    year = filename[-10:-6]
    month = filename[-6:-4]
    return f"Imbalance-{year}-{month}.xls"

def rule_wind(filename):
    if len(filename) > 24:
        year = filename[-21:-17]
        month = filename[-17:-15]
        base = filename[:-17]
        return f"{base}_{year}_{month}.xls"
    else:
        return filename

def rule_pv(filename):
    if len(filename) > 25:
        year = filename[-21:-17]
        month = filename[-17:-15]
        base = filename[:-22]
        return f"{base}_{year}-{month}.xls"
    else:
        return filename

def rule_years(filename):
    # Rule for Elia filenames of historical production
    if len(filename) > 24:
        new_filename = filename[:12] + filename[17:]
        return new_filename
    else:
        return filename

# def rule_point(filename):
#     if ".." in filename:
#         filename = filename.replace("..",".")
#     return filename


if __name__ == "__main__":
    folder_path = "C:/Users/u0137781/OneDrive - KU Leuven/data/SI_forecasting/Elia/RES/Wind Total"  # Folder path for file renaming

    target_rules = {
        #Dict with keys being substring of filenames that trigger a specific rule, being the value associated to that key
        "MonthlyCIPUGenerationPerFuelTypeByQuarterHour": rule_gen_prod,
        #"Generation_Produced_Historical": rule_point,
        "ImbalanceNrvPrices": rule_imb_nrv,
        'Generation_Forecast_Historical_Cipu': rule_gen_fc,
        "WindForecast": rule_wind,
        "SolarForecast": rule_pv
    }

    rename_files_in_folder(folder_path,target_rules) #Call rename function



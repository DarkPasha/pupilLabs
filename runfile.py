'''
Program runfile to export data from Pupil Labs recordings to CSV and HDF5 formats. referencing lib/getData.py
hdf5 files can be read again as a pandas DataFrame.An example of how to do this is in examples/extractHDF5.py



'''

import lib.getData as getData
import json

# Load config file
with open("config.json", "r") as f:
    
    config = json.load(f)

# Extract parameters
recording_file = config["recording_file"]
recording_number = config["recording_number"]
export_path = config["export_path"]
export_csv = config.get("csv", True)
export_hdf5 = config.get("hdf5", True)

# Call export
getData.export(csv=export_csv, hdf5=export_hdf5, recording_file=recording_file, recording_number=recording_number, export_path=export_path, resample=True)



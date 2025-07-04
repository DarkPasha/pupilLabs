import pandas as pd
from pathlib import Path


def resample_pupillabs(hdf5_file_path, df_key : str, resampling_frequency: int,logging=None):
    """
    Resample the data in a Pupil Labs HDF5 file to a specified frequency and save it as a CSV file.
    
    :param hdf5_file_path: Path to the HDF5 file containing Pupil Labs data.
    :param df_key: Key in the HDF5 file to access the DataFrame. (Keys in the HDF5 file: ['blinks', 'events', 'eyestates', 'fixations', 'gaze', 'imu', 'saccades'])
    :param resampling_frequency: Frequency in Hz to which the data should be resampled.
    :param logging: Logger object to log messages.
    """
    import logging

    if logging is None:

        log_dir = Path("log")
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(),  
                logging.FileHandler(rf"log\{resampling_frequency}hz_export.log")  
            ]
        )

    export_path=Path(hdf5_file_path).parent

    resampling_folder_path = export_path / f"resampled_{resampling_frequency}Hz"
    resampling_folder_path.mkdir(parents=True, exist_ok=True)

    resampling_csv_folder_path = resampling_folder_path / "csv"
    resampling_csv_folder_path.mkdir(parents=True, exist_ok=True)
    resampled_file_hdf5_path = resampling_folder_path / f"{Path(hdf5_file_path).stem}_resampled_{resampling_frequency}Hz.hdf5"
    
    resampled_file_csv_path = resampling_csv_folder_path / f"{df_key}.csv"

    resampling_time=(1/resampling_frequency)*1000  # Convert Hz to milliseconds

    resampling_time= str(resampling_time) + "ms"
    df = pd.read_hdf(hdf5_file_path, key=df_key)
    df["timestamp [ns]"] = pd.to_datetime(df["timestamp [ns]"],unit="ns")
    df = df.set_index("timestamp [ns]")

    time_diff = df.index.to_series().diff().dropna()
    avg_interval_seconds = time_diff.mean().total_seconds()
    raw_sampling_frequency = 1 / avg_interval_seconds

    logging.info(f"Estimated Original Sampling Frequency for {df_key}: {raw_sampling_frequency:.2f} Hz")

    try :
        df.drop("recording id", axis=1, inplace=True)
    except Exception as e:
        logging.warning(f"Could not delete column 'recording id': {e}")
    try:
        df.drop("fixation id", axis=1, inplace=True)
    except Exception as e:
        logging.warning(f"Could not delete column 'fixation id': {e}")
    try:
        df.drop("blink id", axis=1, inplace=True)
    except Exception as e:
        logging.warning(f"Could not delete column 'blink id': {e}")

    df_resampled = df.resample(resampling_time).mean()
    df_resampled.to_hdf(resampled_file_hdf5_path, key=df_key, mode='a')

    df_resampled.to_csv(resampled_file_csv_path, index=True)



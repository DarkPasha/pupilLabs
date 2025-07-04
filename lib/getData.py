'''
This script exports data from a Pupil Labs Neon recording into various formats such as CSV and HDF5.
It includes functions to export gaze data, blinks, fixations, saccades, eye states, IMU data, events, recording info, scene camera calibration, and world timestamps.
'''

import json

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


from scipy.spatial.transform import Rotation

import pupil_labs.neon_recording as nr

import logging


def setup_logging():
    """
    
    Sets up logging configuration. Create a log directory if it does not exist, and configure logging to output to both the console and a file.
    
    
    """
    log_dir = Path("log")
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  
            logging.FileHandler(r"log\export.log")  
        ]
    )

def unproject_points(points_2d, camera_matrix, distortion_coefs, normalize=False):
    """Undistorts points according to the camera model.

    :param pts_2d, shape: Nx2
    :return: Array of unprojected 3d points, shape: Nx3
    """
    # Convert type to numpy arrays (OpenCV requirements)
    camera_matrix = np.array(camera_matrix)
    distortion_coefs = np.array(distortion_coefs)
    points_2d = np.asarray(points_2d, dtype=np.float32)

    # Add third dimension the way cv2 wants it
    points_2d = points_2d.reshape((-1, 1, 2))

    # Undistort 2d pixel coordinates
    points_2d_undist = cv2.undistortPoints(points_2d, camera_matrix, distortion_coefs)
    # Unproject 2d points into 3d directions; all points. have z=1
    points_3d = cv2.convertPointsToHomogeneous(points_2d_undist)
    points_3d.shape = -1, 3

    if normalize:
        # normalize vector length to 1
        points_3d /= np.linalg.norm(points_3d, axis=1)[:, np.newaxis]

    return points_3d


def cart_to_spherical(points_3d, apply_rad2deg=True):
    
    points_3d = np.asarray(points_3d)
    # convert cartesian to spherical coordinates
    # source: http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    # elevation: vertical direction
    #   positive numbers point up
    #   negative numbers point bottom
    elevation = np.arccos(y / radius) - np.pi / 2
    # azimuth: horizontal direction
    #   positive numbers point right
    #   negative numbers point left
    azimuth = np.pi / 2 - np.arctan2(z, x)

    if apply_rad2deg:
        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)

    return radius, elevation, azimuth


def find_ranged_index(values, left_boundaries, right_boundaries):
    left_ids = np.searchsorted(left_boundaries, values, side="right") - 1
    right_ids = np.searchsorted(right_boundaries, values, side="right")

    return np.where(left_ids == right_ids, left_ids, -1)


def export_gaze(recording, export_path,csv: bool = True, hdf5: bool = True, hdf5_path= None):
    """
    Exports gaze data from a recording to CSV and/or HDF5 format.
    
    
    :param recording: The recording object containing gaze data.
    :param export_path: The path where the exported files will be saved.
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param hdf5_path: The path for the HDF5 file if exporting to HDF5.
    :return: None
    
    """

    fixations = recording.fixations[recording.fixations["event_type"] == 1]

    fixation_ids = (
        find_ranged_index(recording.gaze.ts, fixations.start_ts, fixations.end_ts) + 1
    )

    blink_ids = (
        find_ranged_index(
            recording.gaze.ts, recording.blinks.start_ts, recording.blinks.end_ts
        )
        + 1
    )

    spherical_coords = cart_to_spherical(
        unproject_points(
            recording.gaze.xy,
            recording.calibration.scene_camera_matrix,
            recording.calibration.scene_distortion_coefficients,
        )
    )

    gaze = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.gaze.ts,
        "gaze x [px]": recording.gaze.x,
        "gaze y [px]": recording.gaze.y,
        "worn": recording.worn.worn,
        "fixation id": fixation_ids,
        "blink id": blink_ids,
        "azimuth [deg]": spherical_coords[2],
        "elevation [deg]": spherical_coords[1],
    })

    gaze_dict=gaze.to_dict(orient='records')

    gaze["fixation id"] = gaze["fixation id"].replace(0, None)
    gaze["blink id"] = gaze["blink id"].replace(0, None)

    if csv:
        try:
            export_file = export_path / "gaze.csv"
            gaze.to_csv(export_file, index=False)
            logging.info(f"Wrote {export_file}")
        except Exception as e:
            logging.error(f"An error occurred while writing gaze data to CSV: {e}")
            return
           


    if hdf5:
        try:

            gaze.to_hdf(hdf5_path, key="gaze", mode="a" )
            logging.info(f"Wrote gaze in {hdf5_path}")

        except Exception as e:

            logging.error(f"An error occurred while writing gaze data to HDF5: {e}")

            return


def export_blinks(recording, export_path,csv: bool = True, hdf5: bool = True, hdf5_path= None):
    """
    Exports blink data from a recording to CSV and/or HDF5 format.


    :param recording: The recording object containing blink data.
    :param export_path: The path where the exported files will be saved.
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param hdf5_path: The path for the HDF5 file if exporting to HDF5.
    :return: None
    
    """
    blinks = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "blink id": 1 + np.arange(len(recording.blinks)),
        "start timestamp [ns]": recording.blinks.start_ts,
        "end timestamp [ns]": recording.blinks.end_ts,
        "duration [ms]": (recording.blinks.end_ts - recording.blinks.start_ts) / 1e6,
    })
    if csv:
        try:
            export_file = export_path / "blinks.csv"
            blinks.to_csv(export_file, index=False)
            print(f"Wrote {export_file}")
        except Exception as e:
            logging.error(f"An error occurred while writing blinks data to CSV: {e}")
            return

    
    if hdf5:
        try:

            blinks.to_hdf(hdf5_path, key="blinks", mode="a" )
            logging.info(f"Wrote blinks in {hdf5_path}")

        except Exception as e:

            logging.error(f"An error occurred while writing blinks data to HDF5: {e}")
            
            return


def export_fixations(recording, export_path,csv: bool = True, hdf5: bool = True, hdf5_path= None):
    """
    Exports fixation data from a recording to CSV and/or HDF5 format.


    :param recording: The recording object containing fixation data.
    :param export_path: The path where the exported files will be saved.
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param hdf5_path: The path for the HDF5 file if exporting to HDF5.
    :return: None
    """
    fixations_only = recording.fixations[recording.fixations["event_type"] == 1]

    spherical_coords = cart_to_spherical(
        unproject_points(
            fixations_only.mean_gaze_xy,
            recording.calibration.scene_camera_matrix,
            recording.calibration.scene_distortion_coefficients,
        )
    )

    fixations = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "fixation id": 1 + np.arange(len(fixations_only)),
        "start timestamp [ns]": fixations_only.start_ts,
        "end timestamp [ns]": fixations_only.end_ts,
        "duration [ms]": (fixations_only.end_ts - fixations_only.start_ts) / 1e6,
        "fixation x [px]": fixations_only.mean_gaze_xy[:, 0],
        "fixation y [px]": fixations_only.mean_gaze_xy[:, 1],
        "azimuth [deg]": spherical_coords[2],
        "elevation [deg]": spherical_coords[1],
    })

    if csv:
        try:
            export_file = export_path / "fixations.csv"
            fixations.to_csv(export_file, index=False)
            print(f"Wrote {export_file}")
        except Exception as e:
            logging.error(f"An error occurred while writing fixations data to CSV: {e}")
            return

    
    if hdf5:
        try:

            fixations.to_hdf(hdf5_path, key="fixations", mode="a" )
            logging.info(f"Wrote fixations in {hdf5_path}")

        except Exception as e:

            logging.error(f"An error occurred while writing fixations data to HDF5: {e}")
            
            return


def export_saccades(recording, export_path,csv: bool = True, hdf5: bool = True, hdf5_path= None):
    """
    Exports saccade data from a recording to CSV and/or HDF5 format.


    :param recording: The recording object containing saccade data.
    :param export_path: The path where the exported files will be saved.
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param hdf5_path: The path for the HDF5 file if exporting to HDF5.
    :return: None
    """
    saccades_only = recording.fixations[recording.fixations["event_type"] == 0]

    saccades = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "saccade id": 1 + np.arange(len(saccades_only)),
        "start timestamp [ns]": saccades_only.start_ts,
        "end timestamp [ns]": saccades_only.end_ts,
        "duration [ms]": (saccades_only.end_ts - saccades_only.start_ts) / 1e6,
        "amplitude [px]": saccades_only.amplitude_pixels,
        "amplitude [deg]": saccades_only.amplitude_angle_deg,
        "mean velocity [px/s]": saccades_only.mean_velocity,
        "peak velocity [px/s]": saccades_only.max_velocity,
    })

    if csv:
        try:
            export_file = export_path / "saccades.csv"
            saccades.to_csv(export_file, index=False)
            print(f"Wrote {export_file}")
        except Exception as e:
            logging.error(f"An error occurred while writing saccades data to CSV: {e}")
            return

    if hdf5:
        try:

            saccades.to_hdf(hdf5_path, key="saccades", mode="a" )
            logging.info(f"Wrote saccades in {hdf5_path}")

        except Exception as e:

            logging.error(f"An error occurred while writing saccades data to HDF5: {e}")
            
            return


def export_eyestates(recording, export_path,csv: bool = True, hdf5: bool = True, hdf5_path= None):
    """
    Exports eye state data from a recording to CSV and/or HDF5 format.


    :param recording: The recording object containing eye state data.
    :param export_path: The path where the exported files will be saved.
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param hdf5_path: The path for the HDF5 file if exporting to HDF5.
    :return: None
    
    """
    es = recording.eye_state
    eyestates = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": es.ts,
        "pupil diameter left [mm]": es.pupil_diameter_left_mm,
        "pupil diameter right [mm]": es.pupil_diameter_right_mm,
        "eyeball center left x [mm]": es.eyeball_center_left_xyz[:, 0],
        "eyeball center left y [mm]": es.eyeball_center_left_xyz[:, 1],
        "eyeball center left z [mm]": es.eyeball_center_left_xyz[:, 2],
        "eyeball center right x [mm]": es.eyeball_center_right_xyz[:, 0],
        "eyeball center right y [mm]": es.eyeball_center_right_xyz[:, 1],
        "eyeball center right z [mm]": es.eyeball_center_right_xyz[:, 2],
        "optical axis left x": es.optical_axis_left_xyz[:, 0],
        "optical axis left y": es.optical_axis_left_xyz[:, 1],
        "optical axis left z": es.optical_axis_left_xyz[:, 2],
        "optical axis right x": es.optical_axis_right_xyz[:, 0],
        "optical axis right y": es.optical_axis_right_xyz[:, 1],
        "optical axis right z": es.optical_axis_right_xyz[:, 2],
        "eyelid angle top left [rad]": es.eyelid_angle[:, 0],
        "eyelid angle bottom left [rad]": es.eyelid_angle[:, 1],
        "eyelid aperture left [mm]": es.eyelid_aperture_left_right_mm[:, 0],
        "eyelid angle top right [rad]": es.eyelid_angle[:, 2],
        "eyelid angle bottom right [rad]": es.eyelid_angle[:, 3],
        "eyelid aperture right [mm]": es.eyelid_aperture_left_right_mm[:, 1],
    })

    if csv:
        try:
            export_file = export_path / "3d_eye_states.csv"
            eyestates.to_csv(export_file, index=False)
            print(f"Wrote {export_file}")
        except Exception as e:
            logging.error(f"An error occurred while writing eyestates data to CSV: {e}")
            return

    if hdf5:
        try:

            eyestates.to_hdf(hdf5_path, key="eyestates", mode="a" )
            logging.info(f"Wrote eyestates in {hdf5_path}")

        except Exception as e:

            logging.error(f"An error occurred while writing eyestates data to HDF5: {e}")
            
            return


def export_imu(recording, export_path,csv: bool = True, hdf5: bool = True, hdf5_path= None):
    """
    Exports IMU data from a recording to CSV and/or HDF5 format.


    :param recording: The recording object containing IMU data.
    :param export_path: The path where the exported files will be saved.
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param hdf5_path: The path for the HDF5 file if exporting to HDF5.
    :return: None
    """
    rotations = Rotation.from_quat(recording.imu.quaternion_wxyz, scalar_first=True)
    eulers = rotations.as_euler(seq="yxz", degrees=True)

    imu = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.imu.ts,
        "gyro x [deg/s]": recording.imu.gyro_xyz[:, 0],
        "gyro y [deg/s]": recording.imu.gyro_xyz[:, 1],
        "gyro z [deg/s]": recording.imu.gyro_xyz[:, 2],
        "acceleration x [g]": recording.imu.accel_xyz[:, 0],
        "acceleration y [g]": recording.imu.accel_xyz[:, 1],
        "acceleration z [g]": recording.imu.accel_xyz[:, 2],
        "roll [deg]": eulers[:, 0],
        "pitch [deg]": eulers[:, 1],
        "yaw [deg]": eulers[:, 2],
        "quaternion w": recording.imu.quaternion_wxyz[:, 0],
        "quaternion x": recording.imu.quaternion_wxyz[:, 1],
        "quaternion y": recording.imu.quaternion_wxyz[:, 2],
        "quaternion z": recording.imu.quaternion_wxyz[:, 3],
    })

    if csv:
        try:
            export_file = export_path / "imu.csv"
            imu.to_csv(export_file, index=False)
            print(f"Wrote {export_file}")
        except Exception as e:
            logging.error(f"An error occurred while writing imu data to CSV: {e}")
            return

    if hdf5:
        try:

            imu.to_hdf(hdf5_path, key="imu", mode="a" )
            logging.info(f"Wrote imu in {hdf5_path}")

        except Exception as e:

            logging.error(f"An error occurred while writing imu data to HDF5: {e}")
            
            return


def export_events(recording, export_path,csv: bool = True, hdf5: bool = True, hdf5_path= None):
    """
    Exports event data from a recording to CSV and/or HDF5 format.


    :param recording: The recording object containing event data.
    :param export_path: The path where the exported files will be saved.
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param hdf5_path: The path for the HDF5 file if exporting to HDF5.
    :return: None
    """
    events = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.events.ts,
        "name": recording.events.event,
        "type": "recording",
    })

    if csv:
        try:
            export_file = export_path / "events.csv"
            events.to_csv(export_file, index=False)
            print(f"Wrote {export_file}")
        except Exception as e:
            logging.error(f"An error occurred while writing events data to CSV: {e}")
            return

    if hdf5:
        try:

            events.to_hdf(hdf5_path, key="events", mode="a" )
            logging.info(f"Wrote events in {hdf5_path}")

        except Exception as e:

            logging.error(f"An error occurred while writing events data to HDF5: {e}")
            
            return


def export_info(recording, export_path,csv: bool = True,hdf5: bool = True, hdf5_path= None):
    """
    Exports recording information to a JSON file.
    
    :param recording: The recording object containing information.
    :param export_path: The path where the exported file will be saved.

    """

    with (export_path / "info.json").open("w") as f:
        json.dump(recording.info, f, indent=4, sort_keys=True)


def export_scene_camera_calibration(recording, export_path,csv: bool = True,hdf5: bool = True, hdf5_path= None):
    """
    Exports scene camera calibration data to a JSON file.


    :param recording: The recording object containing camera calibration data.
    :param export_path: The path where the exported file will be saved.
    
    """
    distortion = recording.calibration.scene_distortion_coefficients.reshape([1, -1])
    camera_info = {
        "camera_matrix": recording.calibration.scene_camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "serial_number": recording.calibration.serial,
    }
    with (export_path / "scene_camera.json").open("w") as f:
        json.dump(camera_info, f, indent=4, sort_keys=True)


def export_world_timestamps(recording, export_path,csv: bool = True,hdf5: bool = True, hdf5_path= None):
    """
    Exports world timestamps from a recording to a CSV file.
    :param recording: The recording object containing world timestamps.
    :param export_path: The path where the exported file will be saved.
    """
    events = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.scene.ts,
    })

    export_file = export_path / "world_timestamps.csv"
    events.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")

def export(csv: bool =True, hdf5: bool = True, recording_file: str = None, recording_number: str = None,export_path: str = None, resample: bool = False):
    """
    main callable function to export data from a Pupil Labs recording.

    
    :param csv: Boolean indicating whether to export to CSV format.
    :param hdf5: Boolean indicating whether to export to HDF5 format.
    :param recording_file: The path to the recording file. If None, defaults to the parent directory of the current working directory.
    :param recording_number: The recording number. If None, defaults to the first directory in the recording file path.
    :param export_path: The path where the exported files will be saved. If None, defaults to a directory named "export" in the parent directory of the current working directory.
    :return: None

    """
    
    setup_logging()

    try: # check if recording file and number are provided, if not, use default values

        if recording_file is None:
            current_dir = Path.cwd()
            parent_dir = current_dir.parent
            recording_file = parent_dir / 'testDataset'
            logging.warning(f'No recording file path provided. using {parent_dir} as default recording file path.')
        else:
            logging.info(f'Using {recording_file} as recording file path.')
            recording_file = Path(recording_file)

        if recording_number is None:
            recording_number = next(recording_file.iterdir())
            logging.warning(f'No recording number provided. using {recording_number} as default recording number.')
        else:
            logging.info(f'Using {recording_number} as recording number.')
            recording_number = Path(recording_number)


        recording_path = recording_file / recording_number
       
    except Exception as e:

        logging.error(f"An error occurred while getting the recording file or number: {e}, recording directory: {recording_path}")
        
        return
    
    try: # check if export path is provided, if not, use default values
        if export_path is None:
            current_dir = Path.cwd()
            parent_dir = current_dir.parent
            export_path = parent_dir / Path("export")
            export_path.mkdir(parents=True, exist_ok=True)

            logging.warning(f'No export path provided. using {export_path} as default export path. Making new directory.')
        
        else:
            logging.info(f'Using {export_path} as export path.')
            export_path = Path(export_path)
    
    except Exception as e:
        logging.error(f"An error occurred while getting the export path: {e}, export path: {export_path}")
        return


    try: # open the recording file
        recording = nr.open(recording_path)

    except Exception as e:

        logging.error(f"An error occurred while opening the recording: {e}, recording path: {recording_path}, export path: {export_path}")
        
        return
    
    
    func_map = {
        "gaze": export_gaze,
        "blinks": export_blinks,
        "fixations": export_fixations,
        "saccades": export_saccades,
        "eyestates": export_eyestates,
        "imu": export_imu,
        "events": export_events,
        "info": export_info,
        "scene-camera": export_scene_camera_calibration,
        "world-timestamps": export_world_timestamps,
    }

    
    export_path= export_path / recording_number
    export_path.mkdir(parents=True, exist_ok=True)

    data_keys = [f"--{k}" for k in func_map]
    if hdf5:
        recording_number_hdf5 = str(recording_number) + '.hdf5'
        hdf5_file = str(export_path / recording_number_hdf5)

    if csv:
        export_path= export_path / 'csv'
        export_path.mkdir(parents=True, exist_ok=True)

        

    for stream_name, export_func in func_map.items():
        if f"--{stream_name}" in data_keys:
            export_func(recording, export_path,csv=csv, hdf5=hdf5, hdf5_path=hdf5_file)


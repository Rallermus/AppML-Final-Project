from pathlib import Path
from typing import Optional, Union
import numpy as np

import pandas as pd

COLUMN_MAP = {
    # time
    "Year": "year",
    "QAR YEAR": "year",
    "Date: Year (Derived)": "year",

    "MONTH": "month",
    "QAR MONTH": "month",
    "Date (month)": "month",

    "DAY": "day",
    "QAR DAY": "day",
    "Date (day)": "day",

    "GMT - Hours (BCD)": "hour",
    "QAR HOUR": "hour",
    "UTC Hours": "hour",

    "GMT - Minutes (BCD)": "minute",
    "QAR MINUTE": "minute",
    "UTC Minutes": "minute",

    "GMT Seconds": "second",
    "QAR SECOND": "second",
    "UTC Seconds": "second",

    # altitude
    "Radio Altitude": "radio_altitude",
    "Radio Height 1": "radio_altitude",

    # vertical acceleration
    "Vertical Acceleration": "vert_acc",
    "Normal acceleration": "vert_acc",
}



def import_flight_data(
    data_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.csv",
    recursive: bool = True,
    start: Optional[int] = 0,
    limit: Optional[int] = None,
    add_source_file: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Import CSV flight data from the turbulens data folder.

    Args:
        data_dir: Directory to search. Defaults to `<project_root>/data`.
        pattern: File match pattern, default is '*.csv'.
        recursive: If True, search subfolders recursively.
        start: Optional index to start loading files from.
        limit: Optional max number of files to load.
        add_source_file: If True, add a `source_file` column.
        **read_csv_kwargs: Extra keyword arguments passed to `pd.read_csv`.

    Returns:
        A concatenated pandas DataFrame containing all loaded CSV files.
    """
    root_data_dir = Path(__file__).resolve().parents[1] / "data"
    target_dir = Path(data_dir) if data_dir else root_data_dir

    finder = "rglob" if recursive else "glob"

    arrivals_dir = target_dir / "Arrivals"
    departures_dir = target_dir / "Departures"

    arrivals = sorted(getattr(arrivals_dir, finder)(pattern))
    departures = sorted(getattr(departures_dir, finder)(pattern))
    csv_files = arrivals + departures


    if limit is not None:
        csv_files = csv_files[start:start + limit]
    else:
        csv_files = csv_files[start:]

    if not csv_files:
        raise FileNotFoundError(
            f"No files found in '{target_dir}' using pattern '{pattern}'."
        )

    frames = []
    for csv_path in csv_files:
        if 'GKN' in csv_path.name: #Skip files with OY-GKN, as their format sucks
            continue
        if 'Dep_' in csv_path.name: #Skip departure files, as they are not relevant for the landing prediction task
            continue

        frame = pd.read_csv(csv_path, low_memory=False)

        frame.columns = [COLUMN_MAP.get(c, c) for c in frame.columns]       
        needed = [
            'year','month','day',
            'hour','minute','second',
            'vert_acc','radio_altitude']      
          
        frame = frame[[c for c in needed if c in frame.columns]]        
        mask_time = np.isclose(frame['second'] % 1, 0) #!!!
        mask_alt = frame['radio_altitude'] <= 1500 #!!!
        frame = frame.loc[mask_time & mask_alt].reset_index(drop=True) #!!!

        if add_source_file:
            frame["source_file"] = str(csv_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def extract_wind_data(flight_df, data_dir, offset=10): 
    """
    Extracts wind data from the TWI dataset for a given flight, based on the landing time and an offset for the recording start time.
    Args:
        flight_df (pd.DataFrame): A DataFrame containing data for one flight, including landing time information.
        data_dir (str): The directory containing the wind data files.
        offset (int): The number of minutes before the landing time to start recording wind data. Default is 10 minutes.
    Returns:
        pd.DataFrame: A DataFrame containing the wind data for the specified time range.
    """

    first = flight_df.iloc[0]

    month = first['month'].astype(int)
    day = first['day'].astype(int)
    if day < 10:
        day_string = '0'+str(day)
    else:
        day_string = str(day)

    if month < 10:
        month_string = '0'+str(month)
    else:        
        month_string = str(month)

    year = first['year'].astype(int)
    start_h = first['hour'].astype(int)
    start_m = first['minute'].astype(int)
    start_s = first['second'].astype(int)
    
    start_of_landing_time = pd.Timestamp(year=year,month=month,day=day,hour=start_h,minute=start_m,second=start_s)

    start_of_recording_time = start_of_landing_time - pd.Timedelta(minutes=offset)
    
    if start_of_recording_time.day != start_of_landing_time.day:
        raise ValueError("Start of recording time is on a different day than the landing time. Please adjust the offset or check the flight data.")
    
    date_string = str(year)+'-'+month_string+'-'+day_string
    file_string = 'TWI-'+date_string+'_UTC_log.csv'

    wind_df = pd.read_csv(data_dir + '/' + file_string, sep=';')

    wind_df['DateTime'] = pd.to_datetime(wind_df['DateTime'], format='%Y-%m-%d %H:%M:%S')

    mask_time = (wind_df['DateTime'] >= start_of_recording_time) & (wind_df['DateTime'] <= start_of_landing_time)

    wind_df = wind_df.loc[mask_time].reset_index(drop=True)
    wind_df['source_file'] = first['source_file']
    return wind_df
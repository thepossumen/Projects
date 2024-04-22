# Run this file to read the csv-files, create dataframes with correct datetime objects and
# correct index, and write the dataframes to pickle-files that can be loaded from jupyter for example. 

# REQUIRES Pandas v2.0 !!!

# REMEMBER to change NROWS-variable to None when running on the whole dataset. Use numbers for testing.

import pandas as pd
from time import time, ctime
from shapely.geometry import LineString, Point
from shapely.wkt import loads # to convert strings to shapely objects
import sys

# Some filepaths
points_file = '../Data/gbg_trajectories_points_2019.csv'
trajs_file = '../Data/gbg_trajectories_2019_crossing.csv'
points_pickle_file = '../Data/points.pkl'
trajs_pickle_file = '../Data/trajectories.pkl'

# Number of rows to read from csv-file. Set to None to read all rows. Default is None.
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else None

def create_points_df(points_file=points_file, ):
    print("Reading points csv, converting timestamps to datetime objects,",
          "and sorting the dataframe on trajectory_sid and timestamp...")
    points_df = pd.read_csv(points_file, parse_dates=['timestamp'], usecols=lambda x: x != 'sid',
                        date_format='%Y-%m-%d %H:%M:%S%z', nrows=NROWS).sort_values(['trajectory_sid', 'timestamp'])
    print("Dataframe with timestamped trajectory points created from csv. Timestamps parsed as datetime objects.")
    print("Adding trajectory_sid as an index,...")
    points_df.set_index([points_df.index, 'trajectory_sid'], drop=False, inplace=True)
    return points_df

def write_points_pickle(points_df, points_pickle_file=points_pickle_file):
    points_df.to_pickle(points_pickle_file)
    print("Dataframe written to pickle-file: {}".format(points_pickle_file))

def create_trajs_df(trajs_file=trajs_file):
    print("Reading trajectory csv, converting timestamps to datetime objects,",
            "and sorting the dataframe on sid...")
    trajs_df = pd.read_csv(trajs_file, parse_dates=['start_time', 'stop_time'],
                date_format='%Y-%m-%d %H:%M:%S%z', nrows=NROWS, index_col='sid').sort_index()
    print("Dataframe with trajectories created from csv. Timestamps parsed as datetime objects.")
    #trajs_df.set_index([trajs_df.index, 'sid'], drop=False, inplace=True)
    return trajs_df

def write_trajs_pickle(trajs_df, trajs_pickle_file=trajs_pickle_file):
    trajs_df.to_pickle(trajs_pickle_file)
    print("Dataframe written to pickle-file: {}".format(trajs_pickle_file))

def remove_2p_200kmh(trajs_df, points_df):
    two_point_trajs = trajs_df.data_points == 2
    high_speed = trajs_df.speed > 200
    traj_sids_to_remove = trajs_df[(two_point_trajs & high_speed)].index.values
    print('traj_sids_to_remove:', traj_sids_to_remove)
    print('len: ', len(traj_sids_to_remove), '...')
    print('dropping points...')
    points_df.drop(traj_sids_to_remove, level='trajectory_sid', inplace=True)
    print('dropping trajectories...')
    trajs_df.drop(traj_sids_to_remove, inplace=True)
    return trajs_df, points_df

def fix_geometry(trajs_df):
    trajs_df['line'] = trajs_df.line.apply(lambda x: loads(x))
    return trajs_df
    

if __name__ == '__main__':
    program_start = time()

    print("NROWS:", NROWS)
    print("Creating dataframe with trajectory geometries.")
    start = time()
    print("Timer started... allow 20-40 seconds...")
    trajs_df = create_trajs_df()
    end = time()
    print("Number of rows handled: ", NROWS if NROWS else trajs_df.shape[0])
    print(f"Time elapsed: {end-start:.2f} seconds.\n")

    cur_mem = int(trajs_df.memory_usage().sum()) / (1024*1024)
    print('Current memory usage: {:.2f} MB'.format(cur_mem))
    print("Casting all integer type columns to integer type uint32 to save memory...")
    start = time()
    trajs_df[['duration','data_points','distance']] = trajs_df[['duration','data_points','distance']].astype('uint32')
    cur_mem = int(trajs_df.memory_usage().sum()) / (1024*1024)
    print('New memory usage: {:.2f} MB'.format(cur_mem))
    end = time()
    print(f"Time elapsed: {end-start:.2f} seconds.\n")

    print("Creating dataframe with trajectory points.")
    start = time()
    print("Timer started... allow for 2-4 minutes... time now: ", ctime(time()))
    points_df = create_points_df()
    end = time()
    print("Number of rows handled: ", NROWS if NROWS else points_df.shape[0])
    print(f"Time elapsed: {end-start:.2f} seconds.\n")

    cur_mem = int(points_df.memory_usage().sum()) / (1024*1024)
    print('Current memory usage: {:.2f} MB'.format(cur_mem))
    print("Casting all integer type columns to integer type uint32 to save memory...")
    start = time()
    points_df[['trajectory_sid']] = points_df[['trajectory_sid']].astype('uint32')
    cur_mem = int(points_df.memory_usage().sum()) / (1024*1024)
    print('New memory usage: {:.2f} MB'.format(cur_mem))
    end = time()
    print(f"Time elapsed: {end-start:.2f} seconds.\n")

    if NROWS == None:
        print("Removing two-point-trajectories with a speed above 200 km/h...")
        start = time()
        print("Timer started...")
        trajs_df, points_df = remove_2p_200kmh(trajs_df, points_df)
        end = time()
        print(f"Time elapsed: {end-start:.2f} seconds.\n")

    print("Fixing geometry...")
    start = time()
    print("Timer started...")
    trajs_df = fix_geometry(trajs_df)
    end = time()
    print(f"Time elapsed: {end-start:.2f} seconds.\n")

    print("Writing trajectory df to pickle-file...")
    write_trajs_pickle(trajs_df)
    print("Writing points df to pickle-file...")
    write_points_pickle(points_df)

    print("Deleting dataframes from memory...")
    del trajs_df
    del points_df


    print("\nDone.")
    program_end = time()
    print("Total program time: {:.2f} seconds.".format(program_end-program_start))


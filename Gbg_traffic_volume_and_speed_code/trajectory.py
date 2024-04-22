import pandas as pd 
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString

class Trajectory():
    def __init__(self, id, df, id_col):
        self.id = id
        self.df = df    
        self.id_col = id_col
         
    def __str__(self):
        return "Trajectory {1} ({2} to {3}) | Size: {0}".format(
            self.df.geometry.count(), self.id, self.get_start_time(), 
            self.get_end_time())
         
    def get_start_time(self):
        return self.df.index.min()
         
    def get_end_time(self):
        return self.df.index.max()
         
    def to_linestring(self):
        return self.make_line(self.df)
         
    def make_line(self, df):
        if df.size > 1:
            return df.groupby(self.id_col)['geometry'].apply(
                lambda x: LineString(x.tolist())).values[0]
        else:
            raise RuntimeError('Dataframe needs at least two points to make line!')
 
    def get_position_at(self, t):
        try:
            return self.df.loc[t]['geometry'][0]
        except:
            return self.df.iloc[self.df.index.drop_duplicates().get_loc(
                t, method='nearest')]['geometry']
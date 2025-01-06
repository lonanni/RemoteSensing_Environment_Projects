import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class UseEnvDigitalTwin:
    '''
    A class to use the Environment (tide+weather) digital twin for Portsmouth Harbour for scientific analysis.

    Args:
    ds (dataframe): complete dataframe - create it with the example notebook or download it from this repository
    year (array of int): year(s) of interest - if not set, covers 2001-2024
    month (array of int): month(s) of interest -if not set, covers all months
    day (array of int): day(s) of interest - if not set, covers all days
    hour (array of int): hour(s) of interest - if not set, covers entire day 
    '''
    def __init__(self, ds, year = np.arange(2001,2024), month = np.arange(1,32), day = np.arange(1,32), hour = np.arange(0,25)):
        self.ds = ds
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        
    def query_columndefintion(self, column_name= []):

        """
        Given a dataset and a column name, returns the definition of that column.

        Parameters:
        delf.ds (dataframe): complete EnvDigitalTwin dataset
        column_name (str): The name of the column for which to retrieve the definition.

        Returns:
        str: The definition of the specified column, or a message if the column doesn't exist.
        
        Flags:
        **inputing more than a string**
        """
        
        ds = self.ds
        column_definitions = {
             'date': 'date when values are collected/computed',
        'ALLSKY_KT' : 'CERES SYN1deg All Sky Insolation Clearness Index (dimensionless) ',
        'ALLSKY_SFC_UVA' : 'CERES SYN1deg All Sky Surface UVA Irradiance (W/m^2)',
        'ALLSKY_SFC_UVB' : 'CERES SYN1deg All Sky Surface UVB Irradiance (W/m^2)',
        'ALLSKY_SFC_UV_INDEX' : 'CERES SYN1deg All Sky Surface UV Index (dimensionless)',
        'T2M' : 'MERRA-2 Temperature at 2 Meters (C)',
        'QV2M' : 'MERRA-2 Specific Humidity at 2 Meters (g/kg)',
        'PRECTOTCORR': 'MERRA-2 Precipitation Corrected (mm/hour)',
        'PS' : 'MERRA-2 Surface Pressure (kPa)',
        'WS10M' : ' MERRA-2 Wind Speed at 10 Meters (m/s)',
        'WD10M' : ' MERRA-2 Wind Direction at 10 Meters (Degrees)',
        'WS50M' : ' MERRA-2 Wind Speed at 50 Meters (m/s)',
        'WD50M' : ' MERRA-2 Wind Direction at 50 Meters (Degrees)',
             'value': 'Surface elevation (unspecified datum) of the water body by bubbler tide gauge',
             'error': 'residual of Surface elevation (unspecified datum) of the water body by bubbler tide gauge',
        }
        
    
        if column_name in list(ds.columns):
            # Return the definition from the dictionary
            return column_definitions.get(column_name, "Definition not found.")
        else:
            return f"Column '{column_name}' not found in the dataset."
        
    def obtain_subset_given_timeframe(self):
        """
        Return DigitalTwin dataset in the time-frame of interest

        Args:
        self.ds (dataframe): complete dataset
        self.year (array of int): year(s) of interest - if not set, covers 2001-2024
        self.month (array of int): month(s) of interest -if not set, covers all months
        self.day (array of int): day(s) of interest - if not set, covers all days
        self.hour (array of int): hour(s) of interest - if not set, covers entire day 
        Returns:
        dataframe subset
        """
        ds = self.ds
        ds["date"] = pd.to_datetime(ds["date"])
        return(ds[(ds["date"].dt.year.isin(self.year))&(ds["date"].dt.month.isin(self.month))\
                  &(ds["date"].dt.day.isin(self.day))&(ds["date"].dt.hour.isin(self.hour))])   
    
    def covariance(self, columns=[]):
        """
        Return covariance amongst the columns of interest

        Args:
        self.ds (dataframe): complate dataset
        columns (array of str): columns of interset. If not set, al columns are considered. 
        self.year (array of int): year(s) of interest
        slf.month (array of int): month(s) of interest
        self.day (array of int): day(s) of interest
        self.hour (array of int): hour(s) of interest, if not set, entire day is returned

        Returns:
        plot covariance matrix
        
        Flags:
        ** at least three columns required **
        
        """
        if columns == []:

            columns = self.ds.drop("date", axis=1).columns
        
        if len(columns)<3:
            return f"input at least 3 column"
        else:
            ds = self.obtain_subset_given_timeframe()[columns]
            corrmat = ds.astype(float).corr().dropna(how="all", axis="columns").dropna(how="all", axis="rows")
            f, ax = plt.subplots( figsize=(15, 15))
            fig = sns.heatmap(corrmat, vmax=1, square=True, cmap="RdBu",  annot=True, vmin=-1);

            return plt.show()
    
    def time_series(self, column ):
        """
        Return time vs column of interest plot, in the timeframe of interest

        Args:
        self.ds (dataframe): complate dataset
        column (str): column of interset.
        self.year (array of int): year(s) of interest
        self.month (array of int): month(s) of interest
        self.day (array of int): day(s) of interest
        self.hour (array of int): hour(s) of interest, if not set, entire day is returned

        Returns:
        plot time vs column of interest
        """
        plt.plot(self.ds.date, self.ds[column])
        plt.xlabel("time")
        plt.ylabel(column)
        plt.show()



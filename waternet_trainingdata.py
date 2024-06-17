import ee
import glob
import os
import geemap.foliumap as emap
import rasterio
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
logging.getLogger('rasterio').setLevel(logging.ERROR)

from pyproj import Geod

class EarthEngineDownloader:
  """
  Initialize the EarthEngineDownloader class.

  Args:
  product (str): satellite data to download (e.g. "LANDSAT/LC08/C02/T1_L2")
  bands (str): band of interest (e.g. ['SR_B2', 'SR_B3'])
  path: Path to the directory where the data will be stored.
  MAX_CLOUD_COVER(int): Maximum cloud cover percentage. Defaults to 10.

  Methods:

  download
  obtain_collection
  download_collection

  """
    # attributes of the class
  def __init__(self, product, bands, path, MAX_CLOUD_COVER=10):
    self.product = product
    self.bands = bands
    self.path = path
    self.MAX_CLOUD_COVER = MAX_CLOUD_COVER

  # methods of the class

  def download(self, toplat, toplong, botlat, botlong, start_year, end_year):
    """
      Defines the time, spatial region, and resolution of the satellite data to
      download and calls the ``download_collection`` function. The spatial region
      is defined as Geometry using lat, lon coordinates for the minimum and maximum
      corners of the rectangle


      Args:
      toplat (float): latitude of the top left corner of the rectangle
      toplong (float): longitude of the top left corner of the rectangle
      botlat (float): latitude of the bottom right corner of the rectangle
      botlong (float): longitude of the bottom right corner of the rectangle
      start_year (int): first year to consider for dowloading the data
      end_year (int): final year to consider for dowloading the data


    """
    region_on_interest = [toplat, toplong, botlat, botlong]

    self.region =  ee.Geometry.Rectangle(region_on_interest)


    # Defines the desired pixel scale for each image,
    # Set to the native resolution of each satellite.
    # This is for the Landsat8
    self.scale_dict = {
            'LS': 30
        }

    years = [str(i) for i in range(start_year,end_year)]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    date_list = [f"{year}-{month}-01" for year in years for month in months]
    date_list.append("2019-01-01")

    dates = [
            (date_list[i], date_list[i+1]) for i in range(len(date_list)-1)
        ]

    # calling self.download_collection
    for start_date, end_date in dates:
      self.download_collection(
              start_date, end_date
            )


  def obtain_collection(self, start_date, end_date):
    """
      Returns image collections in the time and region requested,
      filtered for the cloud coverage (MAX_CLOUD_COVERAGE attribute).


      Args:
      start_date (str): starting date for dowloading the data
      end_date (str): end date for dowloading the data

      Returns:
      image_collections (dict): collection of images to download


    """

    start_date = ee.Date(start_date)


    end_date = ee.Date(end_date).advance(-1, "day")

    # Filter input collections by desired date range, region and max cloud
    # coverage.
    criteria  = ee.Filter.And(
            ee.Filter.geometry(self.region),
            ee.Filter.date(start_date, end_date)
        )

    LS = ee.ImageCollection(self.product) \
                          .filter(criteria) \
                          .filter(
                              ee.Filter.lt('CLOUD_COVER', self.MAX_CLOUD_COVER)
                              ) \
                          .select(self.bands)

    image_collections = {
            'LS': LS,
            }

    return image_collections


  def download_collection(self, start_date, end_date, overwrite=False,):
    """
      Obtains all image collections defined in the request function (``obtain_collection``),
      creates a subfolder in the base directory for the starting date, creates
      additional subfolders for each image collection

      Args:
      start_date (str): starting date for dowloading the data
      end_date (str): end date for dowloading the data
    """
    # Obtains all image collections defined in the request function for the
    # chosen location and date range, with maximum n% cloud cover.
    image_collections = self.obtain_collection(start_date, end_date,)


    # Creates a subfolder in the base directory for the start date
    out_dir = f'{self.path}/{start_date}'
    if not os.path.isdir(out_dir):
      os.mkdir(out_dir)

    # Iterating through each image collection.
    for collection_name, collection in image_collections.items():
      print(collection_name)
      # Counts the number of images in a collection.
      collection_size = collection.size().getInfo()

      # Skips the image collection if it contains no images.
      if collection_size == 0:
        print('No images in collection, skipping.')
        continue

      # Creates additional subfolders for each image collection.
      collection_dir = f'{out_dir}/{collection_name}'
      if not os.path.isdir(collection_dir):
        os.mkdir(collection_dir)

      # Counts number of .tif files already in image collection subfolder.
      tif_count = len(glob.glob1(collection_dir,"*.tif"))

     # Assumes the download for this collection is already complete and
     # therefore skips, provided the number of .tif files already in
     # chosen directory matches the number of images in the collection
     # and overwrite is set to False.
      if collection_size == tif_count and overwrite == False:
        print('Correct number of .tif files for image collection already in directory, skipping.')
        continue

      # Exports each image in the filtered image collection to
      # geoTIFF format.
    emap.ee_export_image_collection(
                    collection,
                    collection_dir,
                    crs='EPSG:4326',
                    scale=self.scale_dict[collection_name],
                    region=self.region
                )
def mask_from_bitmask(bitmask, mask_type):
    '''
    Converts an earth engine QA bitmask to a boolean array of ``mask_type`` pixels.
    Bitmask array conversion from https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding

    Args:
        bitmask (np.ndarray): ``shape(pix_x, pix_y)``
        mask_type (str): String specifying kind of mask to return. Can be ``water``, ``cloud``, or ``shadow``.

    Returns:
        Boolean mask with shape ``shape(pix_x, pix_y)`` indicating pixels of ``mask_type.
    '''

    idx_dict = {
        "water" : 8,
        "cloud" : 12,
        "shadow" : 11,
    }

    # number of bits to convert bitmask to
    m = 16
    # Function to convert an integer to a string binary representation
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    # Calculte binary representations
    strs = to_str_func(bitmask)
    # Create empty array for the bitmask
    bitmask_bits = np.zeros(list(bitmask.shape) + [m], dtype=np.int8)
    # Iterate over all m  bits
    for bit_ix in range(0, m):
        # Get the bits
      fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        # Store the bits
      bitmask_bits[:, :, bit_ix] = fetch_bit_func(strs).astype("int8")

    # The bitmask is stored in bit 7 (index 15-7=8).
    bool_bitmask = bitmask_bits[:, :, idx_dict[mask_type]] == 1

    return bool_bitmask

def get_water_depth(bgrnir):
    '''
   Computes the depth of the water from an image in 4 bands

    Args:
        bgrnir (array): image in 4 bands (blue, green, red, nir)

    Returns:
        water depth, water mask
    '''
    bgrnir = (bgrnir.astype(np.float32))
    water_mask = mask_from_bitmask(bgrnir[:, :, 3].astype(np.int64), 'water') 

    depth = np.zeros_like(bgrnir[:, :, 0])
    depth[water_mask] = ((np.log(bgrnir[:, :, 0][water_mask])/np.log(bgrnir[:, :, 1][water_mask])))

    return depth, water_mask

def scale_im(reader):
    '''
    Scales the image colours in the different bands
    Args:
        reader (array): image in different bands

    Returns:
        scaled image - RGB image
    '''
    red = reader.read(3)
    green = reader.read(2)
    blue = reader.read(1)


    scale = lambda x : (x*0.0000275) - 0.2


    return np.dstack([scale(red), scale(green), scale(blue)]) * 3


def create_color_map(hex_list):
    '''
    Produces a costum colormap for a RGB image
    Args:
        hex_list (list): list of str with HEX values

    Returns:
        color map
    '''
    num_colors = len(hex_list)
    color_positions = np.linspace(0, 1, num_colors)
    color_map_dict = {'red': [], 'green': [], 'blue': []}

    for color_index, hex_color in enumerate(hex_list):
      rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
      color_map_dict['red'].append((color_positions[color_index], rgb_color[0] / 255, rgb_color[0] / 255))
      color_map_dict['green'].append((color_positions[color_index], rgb_color[1] / 255, rgb_color[1] / 255))
      color_map_dict['blue'].append((color_positions[color_index], rgb_color[2] / 255, rgb_color[2] / 255))

    color_map = LinearSegmentedColormap('custom_color_map', color_map_dict)
    return color_map





import math

def calculate_square_corners(center_latitude, center_longitude, distance_around_center):
    # Convert center latitude and longitude to radians
    center_lat_rad = math.radians(center_latitude)
    center_lon_rad = math.radians(center_longitude)

    # Convert distance to radians of arc length
    distance_km = distance_around_center / 1000  # Convert distance to kilometers
    radius_earth_km = 6371  # Average radius of the Earth in kilometers
    arc_length_rad = distance_km / radius_earth_km

    # Calculate latitude and longitude differences
    delta_lat = math.degrees(arc_length_rad)
    delta_lon = math.degrees(arc_length_rad / math.cos(center_lat_rad))

    # Calculate top left corner coordinates
    top_left_lat = center_latitude + delta_lat / 2
    top_left_lon = center_longitude - delta_lon / 2

    # Calculate bottom right corner coordinates
    bottom_right_lat = center_latitude - delta_lat / 2
    bottom_right_lon = center_longitude + delta_lon / 2

    # Return the coordinates as a tuple
    return (top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon)
class MakeTrainingData_Water:
    '''
    A class to make training data for the water mask and the depth of the sea.
    For info on SR bands https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1574_L8_Data_Users_Handbook-v5.0.pdf
    pag 18 of this handbook
    For info on QA pixel pag. 61 of the handbook
    '''
    def __init__(
        self,
        path,
        toplat,
        toplong,
        botlat,
        botlong,
        start_year,
        end_year,
        SR_PROD='LANDSAT/LC08/C02/T1_L2',
        RAW_PROD='LANDSAT/LC08/C02/T1',
        SR_BANDS=['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
        RAW_BANDS=['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        MAX_CLOUD_COVER=10,
        ):
        '''Initialize the MakeTrainingData_WaterDepth class.

        Args:
        - path (str): Path to the directory where the data will be stored.
        - toplat (float): Latitude of the top left corner of the region of interest.
        - toplong (float): Longitude of the top left corner of the region of interest.
        - botlat (float): Latitude of the bottom right corner of the region of interest.
        - botlong (float): Longitude of the bottom right corner of the region of interest.
        - start_year (int): Starting year for the Landsat imagery.
        - end_year (int): Ending year for the Landsat imagery.
        - SR_PROD (str, optional): Landsat surface reflectance product.
                                   Defaults to 'LANDSAT/LC08/C02/T1_L2'.
        - RAW_PROD (str, optional): Landsat raw product.
                                    Defaults to 'LANDSAT/LC08/C02/T1'.
        - SR_BANDS (list, optional): Bands to select from the surface reflectance product.
                                     Defaults to ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'].
        - RAW_BANDS (list, optional): Bands to select from the raw product.
                                      Defaults to ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'].
        - MAX_CLOUD_COVER (int, optional): Maximum cloud cover percentage. Defaults to 10.
        '''

        # Set up region of interest
        region_of_interest = [toplat, toplong, botlat, botlong]
        self.region =  ee.Geometry.Rectangle(region_of_interest)

        # Set up other params
        self.SR_PROD = SR_PROD
        self.RAW_PROD = RAW_PROD
        self.SR_BANDS = SR_BANDS
        self.RAW_BANDS = RAW_BANDS
        self.MAX_CLOUD_COVER = MAX_CLOUD_COVER
        self.SR_PATH = path + '/SR'
        self.RAW_PATH = path + '/RAW'

        # Make paths if not present
        if not os.path.isdir(self.SR_PATH):
            os.mkdir(self.SR_PATH)
        if not os.path.isdir(self.RAW_PATH):
            os.mkdir(self.RAW_PATH)


        # Make date list
        years = [str(i) for i in range(start_year,end_year)]
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        date_list = [f"{year}-{month}-01" for year in years for month in months]
        date_list.append("2019-01-01")
        self.dates = [(date_list[i], date_list[i+1]) for i in range(len(date_list)-1)]

    def download_data(self):
        '''Download the Landsat imagery data for the specified region of interest and time range.'''
        # Define max water count and best file
        water_max = 0
        self.best_file = ''

        # Iterate through the months
        for start_date, end_date in self.dates:

            # Define paths for each month
            SR_OUT_PATH = f'{self.SR_PATH}/{start_date}'
            if not os.path.isdir(SR_OUT_PATH):
                os.mkdir(SR_OUT_PATH)

            RAW_OUT_PATH = f'{self.RAW_PATH}/{start_date}'
            if not os.path.isdir(RAW_OUT_PATH):
                os.mkdir(RAW_OUT_PATH)


            # Convert the dates to a form earth engine likes
            start_date = ee.Date(start_date)
            end_date = ee.Date(end_date).advance(-1, "day")

            # Filter input collections by desired date range, region and cloud coverage.
            criteria  = ee.Filter.And(
                ee.Filter.geometry(self.region),
                ee.Filter.date(start_date, end_date)
            )

            # Get the surface reflectance collection
            SR = ee.ImageCollection(self.SR_PROD) \
                            .filter(criteria) \
                            .filter(ee.Filter.lt('CLOUD_COVER', self.MAX_CLOUD_COVER)) \
                            .select(self.SR_BANDS)

            # Get the RAW collection
            RAW = ee.ImageCollection(self.RAW_PROD) \
                .filter(criteria) \
                .filter(ee.Filter.lt('CLOUD_COVER', self.MAX_CLOUD_COVER)) \
                .select(self.RAW_BANDS)

            # Export the surface reflectance collection to the SR_OUT_PATH directory
            emap.ee_export_image_collection(
                    SR,
                    SR_OUT_PATH,
                    crs='EPSG:4326',
                    scale=30,
                    region=self.region
                )

            # Export the RAW collection to the RAW_OUT_PATH directory
            emap.ee_export_image_collection(
                    RAW,
                    RAW_OUT_PATH,
                    crs='EPSG:4326',
                    scale=30,
                    region=self.region
                )
  # Here the best file is selected as the file with the highest
  # number of water pixel considering also QA

            # Loop through the files in the surface reflectance folder
            for file in glob.glob(SR_OUT_PATH+'/*'):
                # Open the file
                reader = rasterio.open(file)
                # Get the QA pixels
                qa_pix = reader.read(5)
                # Get the water mask
                water_mask = mask_from_bitmask(qa_pix, 'water')
                # Is the number of water pixels bigger than the current max?
                if water_max < water_mask.sum():
                    # If so, update the water max and the best file
                    water_max = water_mask.sum()
                    self.best_file = file

    def get_training_data_depth(self, LAND_FILL_VALUE=0):
        '''Process the downloaded Landsat imagery data and extract training data
         for sea depth estimation.

        Args:
        - LAND_FILL_VALUE (int, optional): Value to fill for land pixels.
         Defaults to -1.

        Returns:
        - X (ndarray): Input training data.
        - y (ndarray): Target training data.
        '''

        # BUGGY! This throws an error every time hence the try except, but the download works.
        # Download data
        try:
            self.download_data()
        except:
            pass

        # Define readers for the best file
        SR_reader = rasterio.open(self.best_file)
        RAW_reader = rasterio.open(self.best_file.replace('SR', 'RAW'))

        # Get the best image
        self.best_im = scale_im(SR_reader)

        # Extract relevant bands from surface reflectance
        bgrnir = np.dstack([
            SR_reader.read(1),
            SR_reader.read(2),
            SR_reader.read(4),
            SR_reader.read(5)
        ]).astype(np.float32)

        # Get the depth
        depth, mask = get_water_depth(bgrnir)

        ninf = ~(depth == -np.inf)
        nnan = ~np.isnan(depth)
        nall = np.logical_and(nnan, ninf)
        self.full_mask = np.logical_and(nall, mask)

        # Get the water depth

        self.full_depth = np.full_like(depth, LAND_FILL_VALUE)
        self.full_depth[self.full_mask] = depth[self.full_mask]

        # Get the raw data
        X = np.dstack([
            RAW_reader.read(1),
            RAW_reader.read(2),
            RAW_reader.read(3),
            RAW_reader.read(4),
            RAW_reader.read(5),
            RAW_reader.read(6),
        ])

        # Reshape X from (pix_x, pix_y, bands) to (num_pix, bands)
        X = X[self.full_mask]

        # Reshape the depth to get y
        y =  self.full_depth
        y = y[self.full_mask]

        return X, y

    def get_training_data_watermask(self, LAND_FILL_VALUE=0):
        '''Process the downloaded Landsat imagery data and extract training data
         for water mask.

        Args:
        - LAND_FILL_VALUE (int, optional): Value to fill for land pixels.
         Defaults to -1.

        Returns:
        - X (ndarray): Input training data.
        - y (ndarray): Target training data.
        '''

        # BUGGY! This throws an error every time hence the try except, but the download works.
        # Download data
        try:
            self.download_data()
        except:
            pass

        # Define readers for the best file
        SR_reader = rasterio.open(self.best_file)
        RAW_reader = rasterio.open(self.best_file.replace('SR', 'RAW'))

        # Get the best image
        self.best_im = scale_im(SR_reader)

        # Extract relevant bands from surface reflectance
        bgrnir = np.dstack([
            SR_reader.read(1),
            SR_reader.read(2),
            SR_reader.read(4),
            SR_reader.read(5)
        ]).astype(np.float32)

        # Get the depth
        depth, mask = get_water_depth(bgrnir)


        # Get the raw data
        X = np.dstack([
            RAW_reader.read(1),
            RAW_reader.read(2),
            RAW_reader.read(3),
            RAW_reader.read(4),
            RAW_reader.read(5),
            RAW_reader.read(6),
        ])

        # Reshape X from (pix_x, pix_y, bands) to (num_pix, bands)

        # Reshape the depth to get y
        y =  mask


        return X, y


    def plot_depthmask(self,vmin = 0,vmax =1, ax=None, **kwargs):
        '''Plot the sea depth map on top of best img for the specified region of interest based on the processed Landsat imagery data.'''
        # Funky colors
        cmap_list = ['2AE3B4', '29C7B1', '28AAAD', '2671A6', ]
        cmap = create_color_map(cmap_list)

        depth = self.full_depth
        water_mask = self.full_mask
        best_file = self.best_file


        # scaling
        scale = lambda x : (x - x[~np.isnan(x)].min())/(x[~np.isnan(x)].max() - x[~np.isnan(x)].min())
        depth[np.logical_and(depth!=-np.inf, water_mask)] = scale(depth[np.logical_and(depth!=-np.inf, water_mask)])
        depth[~np.logical_and(depth!=-np.inf, water_mask)]  = depth[~np.logical_and(depth!=-np.inf, water_mask)].min()

        # Mask the depth array
        depth = np.ma.masked_array(depth, ~water_mask, fill_value=np.nan)

        if ax is None:
          ax = plt.gca()
        # Plot the depth where the depth is
        d = ax.imshow(depth, cmap='turbo_r',vmin = vmin,vmax = vmax, **kwargs)

        # Add a colorbar
        plt.colorbar(d, fraction=0.036, pad=0.04)

        # Show the plot

    def plot_depth(self,vmin = 0,vmax =1, ax=None, **kwargs):
        '''Plot the sea depth map for the specified region of interest based on the processed Landsat imagery data.'''
        # Funky colors
        cmap_list = ['2AE3B4', '29C7B1', '28AAAD', '2671A6', ]
        cmap = create_color_map(cmap_list)

        depth = self.full_depth
        water_mask = self.full_mask
        best_file = self.best_file

        scale = lambda x : (x - x[~np.isnan(x)].min())/(x[~np.isnan(x)].max() - x[~np.isnan(x)].min())
        depth[np.logical_and(depth!=-np.inf, water_mask)] = scale(depth[np.logical_and(depth!=-np.inf, water_mask)])
        depth[~np.logical_and(depth!=-np.inf, water_mask)]  = depth[~np.logical_and(depth!=-np.inf, water_mask)].min()

        # Mask the depth array
        depth = np.ma.masked_array(depth, ~water_mask, fill_value=np.nan)
        self.im = scale_im(rasterio.open(best_file))
        if ax is None:
          ax = plt.gca()

        # Plot image
        ax.imshow(self.im)

        # Plot the depth where the depth is
        d = ax.imshow(depth, cmap='turbo_r',vmin = vmin,vmax = vmax, **kwargs)

        # Add a colorbar
        plt.colorbar(d, fraction=0.036, pad=0.04)


    def plot_img(self, ax=None, **kwargs):
        '''Plot the best image for the loc and time of interest'''
        # Funky colors
        cmap_list = ['2AE3B4', '29C7B1', '28AAAD', '2671A6', ]
        cmap = create_color_map(cmap_list)

        depth = self.full_depth
        water_mask = self.full_mask
        best_file = self.best_file

        scale = lambda x : (x - x[~np.isnan(x)].min())/(x[~np.isnan(x)].max() - x[~np.isnan(x)].min())
        depth[np.logical_and(depth!=-np.inf, water_mask)] = scale(depth[np.logical_and(depth!=-np.inf, water_mask)])
        depth[~np.logical_and(depth!=-np.inf, water_mask)]  = depth[~np.logical_and(depth!=-np.inf, water_mask)].min()

        # Mask the depth array
        depth = np.ma.masked_array(depth, ~water_mask, fill_value=np.nan)
        self.im = scale_im(rasterio.open(best_file))

        if ax is None:
          ax = plt.gca()
        # Plot image
        ax.imshow(self.im, **kwargs)




    def plot_watermask(self,vmin = 0,vmax =1, ax=None, **kwargs):
        '''Plot the sea mask map for loc and time of interest based on the processed Landsat imagery data.'''
        depth = self.full_depth
        water_mask = self.full_mask
        best_file = self.best_file


        scale = lambda x : (x - x[~np.isnan(x)].min())/(x[~np.isnan(x)].max() - x[~np.isnan(x)].min())

        self.im = scale_im(rasterio.open(best_file))

        if ax is None:
          ax = plt.gca()

        d = ax.imshow(water_mask, cmap="BrBG", vmin = vmin,vmax = vmax, **kwargs)

        # Add a colorbar
        plt.colorbar(d, fraction=0.036, pad=0.04)

ee.Authenticate()

ee.Initialize(project="ee-lorenzananni")

def calculate_square_corners(pointlat, pointlong, pad=0.13456632):
    toplat = pointlat + pad
    toplong = pointlong + pad
    botlat = pointlat - pad
    botlong = pointlong - pad
    return toplat, toplong, botlat, botlong


import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import linregress, binned_statistic

class SimpleScaler:
    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
class Sediment():
    def __init__(self, reader):
      self.reader = reader
      self.swir = reader.read(5)
      self.swir2 = reader.read(6)

      self.nir = reader.read(4)
      self.red = reader.read(3)
      self.green = reader.read(2)
      self.blue = reader.read(1)
      
class Sediment_dg():
    """
    As Sediment, this time, we deglint the signal in the different bands
    """
    def __init__(self, reader, water_mask):
      self.reader = reader

      self.nir = reader.read(4)
      self.red = reader.read(3)
      self.green = reader.read(2)
      self.blue = reader.read(1)

      result_B = linregress(self.nir[:,:100].flatten(), self.blue[:,:100].flatten())
      result_G = linregress(self.nir[:,:100].flatten(), self.green[:,:100].flatten())
      result_R = linregress(self.nir[:,:100].flatten(), self.red[:,:100].flatten())

      self.NIR_dg = self.nir - (self.nir  - np.min(self.nir))

      self.R_dg = self.red-result_R.slope* (self.nir  - np.min(self.nir))
      self.G_dg = self.green-result_G.slope* (self.nir  - np.min(self.nir))
      self.B_dg = self.blue-result_B.slope* (self.nir  - np.min(self.nir))

      self.water_mask = water_mask

    def NDTI(self, ):
      """
     **Normalized Difference Turbidity Index (NDTI)**
     **Formula**: \((Red - Green) / (Red + Green)\)
     **Description**: NDTI leverages the fact that sediment-laden water typically has higher reflectance in the red band and lower in the green band compared to clear water.
     **Application**: Higher NDTI values generally indicate higher turbidity levels, which can be associated with surface sediments.


      """
      NDTI_img = (self.R_dg - self.G_dg) / (self.R_dg + self.G_dg)
      img = np.where(self.water_mask == 1, NDTI_img, np.nan)
      return img

    def turbidity_ratio(self, ):

      """
      Turbidity Ratio
      - **Formula**: \(Red / Green\)
      - **Description**: Similar to NDTI, the Red/Green ratio can be used directly to assess the turbidity. Higher ratios suggest more sediment.
      - **Application**: This ratio is simple and effective in identifying sediment concentrations in water.

      """
      tr_img = self.R_dg / self.G_dg
      img = np.where(self.water_mask == 1, tr_img, np.nan)
      return img

    def NDSSI_nir(self, ):
      """
      ### Normalized Difference Suspended Sediment Index (NDSSI)
        - **Formula**: \((Green - SWIR) / (Green + SWIR)\)
        - **Description**: This index contrasts the reflectance in the green band (where water with sediment has higher reflectance) against the SWIR band (where sediment is more absorbent).
        - **Application**: NDSSI can help differentiate between clear water and water with suspended sediments.

      """
      NDSSI_img = (self.G_dg - self.NIR_dg) / (self.G_dg + self.NIR_dg)
      img = np.where(self.water_mask == 1, NDSSI_img, np.nan)
      return img

    def TSM(self, ):
      """
      ### TOTAL SUSPENDED SEDIMENT (tsm)

      """
      TSM_img = 3957*(((self.G_dg + self.R_dg)*0.001) / 2)*1.6436
      img = np.where(self.water_mask == 1, TSM_img, np.nan)
      return img

    def SPM(self, ):
      """
      ### Suspended Particulate Model

      """
      SPM_ind = 2.26*(self.R_dg/self.G_dg)**3 - 5.42 * (self.R_dg/self.G_dg)**2 + 5.58 * (self.R_dg/self.G_dg) - 0.72
      SPM_img = 10**SPM_ind - 1.43
      img = np.where(self.water_mask == 1, SPM_ind, np.nan)
      return img

    def NDTI(self, prediction):
      """
     - **Normalized Difference Turbidity Index (NDTI)**
     - **Formula**: \((Red - Green) / (Red + Green)\)
     - **Description**: NDTI leverages the fact that sediment-laden water typically has higher reflectance in the red band and lower in the green band compared to clear water.
     - **Application**: Higher NDTI values generally indicate higher turbidity levels, which can be associated with surface sediments.
     - **Paper**: https://www.researchgate.net/figure/Normalized-difference-turbidity-index-NDTI-in-Sentinel-2-images-with-synoptical-clear_fig2_364947274

      """
      NDTI_img = (self.red - self.green) / (self.red + self.green)
      img = np.where(prediction == 1, NDTI_img, np.nan)
      return img

    def turbidity_ratio(self, prediction):

      """
      Turbidity Ratio
      - **Formula**: \(Red / Green\)
      - **Description**: Similar to NDTI, the Red/Green ratio can be used directly to assess the turbidity. Higher ratios suggest more sediment.
      - **Application**: This ratio is simple and effective in identifying sediment concentrations in water.
      https://developers.arcgis.com/python/latest/samples/river-turbidity-estimation-using-sentinel2-data-/#:~:text=Normalized%20difference%20turbidity%20index&text=It%20uses%20the%20phenomenon%20that,of%20red%20spectrum%20also%20increases.
      """
      tr_img = self.red / self.green
      img = np.where(prediction == 1, tr_img, np.nan)
      return img

    def NDSSI_nir(self, prediction):
      """
      ### Normalized Difference Suspended Sediment Index (NDSSI)
        - **Formula**: \((Green - SWIR) / (Green + SWIR)\)
        - **Description**: This index contrasts the reflectance in the green band (where water with sediment has higher reflectance) against the SWIR band (where sediment is more absorbent).
        - **Application**: NDSSI can help differentiate between clear water and water with suspended sediments.
        - **Paper**: https://iopscience.iop.org/article/10.1088/1755-1315/98/1/012058/pdf
      """
      NDSSI_img = (self.green - self.nir) / (self.green + self.nir)
      img = np.where(prediction == 1, NDSSI_img, np.nan)
      return img

    def TSM(self, prediction):
      """
      ### TOTAL SUSPENDED SEDIMENT (tsm)
      - *Description**: Total suspended sediment (TSS) is a water quality parameter that is used to understand sediment transport, aquatic ecosystem health, and engineering problems. The TSS method was designed for the wastewater industry, presumably for samples collected after a settling step at a wastewater treatment facility.
      - **Papers**:https://www.mdpi.com/2076-3417/11/15/7082#:~:text=Total%20suspended%20sediment%20(TSS)%20is,ecosystem%20health%2C%20and%20engineering%20problems. https://caltestlabs.com/analytical-services/priority-pollutants/inorganic-methods/tssandssc/ https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/suspended-sediment
      """
      TSM_img = 3957*(((self.green + self.red)*0.001) / 2)*1.6436
      img = np.where(prediction == 1, TSM_img, np.nan)
      return img

    def SPM(self, prediction):
      """
      ### Suspended Particulate Model
      - **Paper**: https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/suspended-particulate-matter#:~:text=SePM%20is%20a%20complex%20mixture,et%20al.%2C%202021a).
      """
      SPM_ind = 2.26*(self.red/self.green)**3 - 5.42 * (self.red/self.green)**2 + 5.58 * (self.red/self.green) - 0.72
      SPM_img = 10**SPM_ind - 1.43
      img = np.where(prediction == 1, SPM_ind, np.nan)
      return img        

def scale_im(reader):
#This scales the image colours so that it looks nice
    red = reader.read(3)
    green = reader.read(2)
    blue = reader.read(1)
    scale = lambda x : (x*0.0000275) - 0.2
    return np.dstack([scale(red), scale(green), scale(blue)]) * 5
    

def run_functions(path_main_directory, function_to_run, scaler, nn):
    if not os.path.isdir(path_main_directory):
        return "Main directory does not exist."
    prediction_count = 0  # To count number of predictions plotted

    contents = os.listdir(path_main_directory)

    subdirectory = path_main_directory

    for dirpath, dirnames, filenames in os.walk(subdirectory):
        if filenames:
            for filename in filenames:

              file_path = os.path.join(dirpath, filename)
              # Plotting and image processing starts here
              if os.path.isfile(file_path) == True:
                # Limit to 6 predictions per location
                if prediction_count >= 6:
                    break  # Stop plotting more predictions for this location
               
                RAW_reader = rasterio.open(file_path)
                image_shape = RAW_reader.read(1).shape
                raw_image = np.dstack([
                    RAW_reader.read(1),
                    RAW_reader.read(2),
                    RAW_reader.read(3),
                    RAW_reader.read(4),
                    RAW_reader.read(5),
                    RAW_reader.read(6),
                ])

                function_to_run(raw_image, RAW_reader, image_shape, scaler, nn)
                prediction_count += 1
                
def plot_prediction(raw_image, RAW_reader, image_shape, scaler, nn):
  cmap_masks = colors.ListedColormap([ 'lightblue', 'blue', 'white', 'black',])
      
  test_image = scale_im(RAW_reader)

  # Resize the images
  test_image_water = scaler.transform(raw_image.reshape((image_shape[0], image_shape[1], 6)))

  predictions = nn(test_image_water.reshape((np.prod(image_shape), 6)))
  predicted_classes = np.argmax(predictions, axis=1)

  # Plotting
  fig, ax = plt.subplots(1, 2, dpi=150)
  ax[0].imshow(test_image)
  ax[0].title.set_text('RGB Raw image')

  im = ax[1].imshow(predicted_classes.reshape((image_shape[0], image_shape[1])), cmap=cmap_masks, interpolation="none", vmin=0, vmax=3)
  cbar = fig.colorbar(im, pad=0.05, fraction=0.046)
  cbar.set_ticks([0, 1, 2, 3])
  cbar.set_ticklabels(["land", "water", "cloud", "blank"])

  ax[1].title.set_text('Keras BCWL Model')

  plt.show()

  # Close the figure after displaying it
  plt.close()
  
def plot_sediment(raw_image, RAW_reader, image_shape, scaler, nn):

  test_image_water = scaler.transform(raw_image.reshape((image_shape[0], image_shape[1], 6)))

  predictions = nn(test_image_water.reshape((np.product(image_shape), 6)))
  predicted_classes = np.argmax(predictions, axis=1)

  rgb_image  = scale_im(RAW_reader)
  # Plotting
  fig, ax = plt.subplots(2, 3, figsize=(14,6.5))
  ax[0, 0].imshow(rgb_image)
  ax[0, 0].title.set_text('RGB image')

  ####
  TSM_img = Sediment(RAW_reader).TSM(predicted_classes.reshape((image_shape[0], image_shape[1])))

  im0 = ax[0, 1].imshow(TSM_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[0, 1].title.set_text('TSM')

  ####
  NDSSI_img = Sediment(RAW_reader).NDSSI_nir(predicted_classes.reshape((image_shape[0], image_shape[1])))

  im0 = ax[0, 2].imshow(NDSSI_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[0, 2].title.set_text('NDSSI nir')

  ####
  turbidity_ratio_img = Sediment(RAW_reader).turbidity_ratio(predicted_classes.reshape((image_shape[0], image_shape[1])))

  im0 = ax[1, 0].imshow(turbidity_ratio_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[1, 0].title.set_text('turbidity ratio')

  ####


  NDTI_img = Sediment(RAW_reader).NDTI(predicted_classes.reshape((image_shape[0], image_shape[1])))

  im0 = ax[1, 1].imshow(NDTI_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[1, 1].title.set_text('NDTI')

  ####



  ####


  SPM_img = Sediment(RAW_reader).SPM(predicted_classes.reshape((image_shape[0], image_shape[1])))

  im0 = ax[1, 2].imshow(SPM_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[1, 2].title.set_text('SPM')

  ####

  plt.show()

  # Close the figure after displaying it
  plt.close()

def plot_sediment_dg(raw_image, RAW_reader, image_shape, scaler, nn):

  test_image_water = scaler.transform(raw_image.reshape((image_shape[0], image_shape[1], 6)))

  predictions = nn(test_image_water.reshape((np.product(image_shape), 6)))
  predicted_classes = np.argmax(predictions, axis=1)
  water_mask = predictions[:,1]
  water_mask = np.where(water_mask > 0.85, 1, 0)
  rgb_image  = scale_im(RAW_reader)
  # Plotting
  fig, ax = plt.subplots(2, 3, figsize=(12,6.5))
  ax[0, 0].imshow(rgb_image)
  ax[0, 0].title.set_text('RGB image')


  TSM_img = Sediment_dg(RAW_reader, water_mask.reshape((image_shape[0], image_shape[1]))).TSM()

  im0 = ax[0, 1].imshow(TSM_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[0, 1].title.set_text('TSM')

  ####
  NDSSI_img = Sediment_dg(RAW_reader, water_mask.reshape((image_shape[0], image_shape[1]))).NDSSI_nir()

  im0 = ax[0, 2].imshow(NDSSI_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[0, 2].title.set_text('NDSSI nir')

  ####
  turbidity_ratio_img = Sediment_dg(RAW_reader, water_mask.reshape((image_shape[0], image_shape[1]))).turbidity_ratio()

  im0 = ax[1, 0].imshow(turbidity_ratio_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[1, 0].title.set_text('turbidity ratio')

  ####


  NDTI_img = Sediment_dg(RAW_reader, water_mask.reshape((image_shape[0], image_shape[1]))).NDTI()

  im0 = ax[1, 1].imshow(NDTI_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[1, 1].title.set_text('NDTI')

  ####


  SPM_img = Sediment_dg(RAW_reader, water_mask.reshape((image_shape[0], image_shape[1]))).SPM()

  im0 = ax[1, 2].imshow(SPM_img, cmap="YlGnBu_r")
  cbar = fig.colorbar(im0, pad=0.05, fraction=0.046)
  ax[1, 2].title.set_text('SPM')

  ####

  plt.show()

  # Close the figure after displaying it
  plt.close()





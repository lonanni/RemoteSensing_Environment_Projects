import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt


class SimpleScaler:
    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        

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

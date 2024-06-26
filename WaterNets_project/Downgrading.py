import math


class DowngradingSatelliteImages:
    '''
    A class to handle the downgrading of satellite images by computing spatial resolutions and scaling factors.

    Args:
    FoV_w (float): angular diameter of the FoV in degrees along the width
    FoV_h (float): angular diameter of the FoV in degrees along the height
    pixel_w (int): number of pixels along the width
    pixel_h (int): number of pixels along the height
    distance (float): distance of the satellite in km
    initial_pix_FoV (float): the pixel resolution of the current image as spatial resolution in km
    '''

    def __init__(self, FoV_w=73, FoV_h=45, pixel_w=4608, pixel_h=2592, distance=400, initial_pix_FoV=0.03):
        self.FoV_w = FoV_w
        self.FoV_h = FoV_h
        self.pixel_w = pixel_w
        self.pixel_h = pixel_h
        self.distance = distance
        self.initial_pix_FoV = initial_pix_FoV

    def compute_pixelspatialresolution(self):
        """
        Computes the spatial resolution of each pixel in km for a given angular diameter of the FoV in degrees,
        a given number of pixels, and a given distance of the satellite in km.

        Returns:
        tuple: spatial resolution of the pixel in km along the x and y dimensions
        """
        h = math.tan(math.radians(self.FoV_h) / 2) * self.distance * 2  # FoV in km given the distance of the satellite along the y
        w = math.tan(math.radians(self.FoV_w) / 2) * self.distance * 2  # FoV in km given the distance of the satellite along the x
        return w / self.pixel_w, h / self.pixel_h

    def compute_pixelscaler(self):
        """
        Computes the scaling factor for downgrading the image given the initial resolution and the new resolution.

        Returns:
        tuple: scaler to use to downgrade the image along the x and y dimensions
        """
        new_w_spatialresolution, new_h_spatialresolution = self.compute_pixelspatialresolution()
        return int(new_w_spatialresolution / self.initial_pix_FoV), int(new_h_spatialresolution / self.initial_pix_FoV)


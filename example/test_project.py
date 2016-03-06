import numpy as np
import fisheye
import os
import glob
import cv2

base_path = r'.\imgs'
NX, NY = 8, 6


def calc_normalization_map(img_resolution, fov=np.pi/2):
    """
    Calc normalization mapping for a camera
    """
    
    #
    # Create a grid of directions.
    #
    X, Y = np.meshgrid(
        np.linspace(-1, 1, img_resolution),
        np.linspace(-1, 1, img_resolution)
    )
    
    PHI = np.arctan2(Y, X)
    PSI = fov * np.sqrt(X**2 + Y**2)
    mask = PSI <= fov
    PSI[~mask] = fov
    
    z = np.cos(PSI)
    x = np.sin(PSI) * np.cos(PHI)
    y = np.sin(PSI) * np.sin(PHI)
    
    XYZ = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

    return XYZ


def main():
    imgs_paths = glob.glob(os.path.join(base_path, '*.jpg'))

    fe = fisheye.FishEye.load('./calib.dat')
    
    XYZ = calc_normalization_map(img_resolution=101)
    
    img = cv2.imread(imgs_paths[0])

    image_points = fe.projectPoints(XYZ)
    
    for pt in image_points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius=2, color=(0, 0, 0), thickness=-1)
    
    cv2.imshow('undistorted', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
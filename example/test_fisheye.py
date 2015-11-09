import fisheye
import os
import glob
import cv2


base_path = r'G:\amit\leipzig\calibration\intrinsic\cam20'
NX, NY = 9, 6


def main():
    imgs_paths = glob.glob(os.path.join(base_path, '*.jpg'))

    fe = fisheye.FishEye(nx=NX, ny=NY, verbose=True)
    rms, K, D, rvecs, tvecs = fe.calibrate(
        imgs_paths,
        show_imgs=True
    )
    
    fe.save('./calib.dat')
    
    img = cv2.imread(imgs_paths[0])
    
    undist_img = fe.undistort(img, undistorted_size=(800, 800))
    
    cv2.imshow('undistorted', undist_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
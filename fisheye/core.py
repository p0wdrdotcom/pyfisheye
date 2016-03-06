from __future__ import print_function
import numpy as np
import pickle
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import os


def load_model(filename):
    """Load a previosly saved fisheye model."""
    
    return FishEye.load(filename)


class FishEye(object):
    """Wrapper around the opencv fisheye calibration code.
    
    """

    def __init__(
        self,
        nx,
        ny,
        verbose=False
        ):
        
        self._nx = nx
        self._ny = ny
        self._verbose = verbose
        self._K = np.zeros((3, 3))
        self._D = np.zeros((4, 1))
        
    def calibrate(
        self,
        img_paths=None,
        imgs=None,
        update_model=True,
        max_iter=30,
        eps=1e-6,
        show_imgs=False,
        calibration_flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        ):
        """Calibrate a fisheye model."""
        
        assert not ((img_paths is None) and (imgs is None)), 'Either specify imgs or img_paths'
        
        chessboard_model = np.zeros((1, self._nx*self._ny, 3), np.float32)
        chessboard_model[0, :, :2] = np.mgrid[0:self._nx, 0:self._ny].T.reshape(-1, 2)
        
        #
        # Arrays to store the chessboard image points from all the images.
        #
        chess_2Dpts_list = []
        
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        
        if show_imgs:
            cv2.namedWindow('checkboard img', cv2.WINDOW_NORMAL)
            cv2.namedWindow('subpix img', cv2.WINDOW_NORMAL)
            cv2.namedWindow('fail img', cv2.WINDOW_NORMAL)
        
        if img_paths is not None:
            imgs = img_paths
        
        for img_index, img in enumerate(imgs):
            
            if type(img) == str:
                fname = img
                if self._verbose:
                    print('Processing img: %s...' % os.path.split(fname)[1], end="")
            
                #
                # Load the image.
                #
                img = cv2.imread(fname)
            else:
                if self._verbose:
                    print('Processing img: %d...' % img_index, end="")
            
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
        
            #
            # Find the chess board corners
            #
            ret, cb_2D_pts = cv2.findChessboardCorners(
                gray,
                (self._nx, self._ny),
                cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        
            if ret:
                #
                # Was able to find the chessboard in the image, append the 3D points
                # and image points (after refining them).
                #
                if self._verbose:
                    print('OK')
                            
                if show_imgs:
                    #
                    # Draw and display the corners
                    #
                    img = cv2.drawChessboardCorners(
                        img, (self._nx, self._ny),
                        cb_2D_pts,
                        ret
                    )
                    cv2.imshow('checkboard img', img)
                    
                cb_2d_pts_subpix = cv2.cornerSubPix(
                    gray,
                    cb_2D_pts,
                    (3, 3),
                    (-1,-1),
                    subpix_criteria
                )
                #
                # The 2D points are reshaped to (1, N, 2). This is a hack to handle the bug
                # in the opecv python wrapper.
                #
                chess_2Dpts_list.append(cb_2d_pts_subpix.reshape(1, -1, 2))
        
                if show_imgs:
                    #
                    # Draw and display the corners
                    #
                    img = cv2.drawChessboardCorners(
                        img, (self._nx, self._ny),
                        cb_2d_pts_subpix,
                        ret
                    )
                    cv2.imshow('subpix img', img)
                    cv2.waitKey(500)
            else:
                if self._verbose:
                    print('FAIL!')         
                    
                if show_imgs:
                    #
                    # Show failed img
                    #
                    cv2.imshow('fail img', img)
                    cv2.waitKey(500)

        if show_imgs:
            #
            # Clean up.
            #
            cv2.destroyAllWindows()

        N_OK = len(chess_2Dpts_list)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        #
        # Update the intrinsic model
        #
        if update_model:
            K = self._K
            D = self._D
        else:
            K = self._K.copy()
            D = self._D.copy()
 
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                [chessboard_model]*N_OK,
                chess_2Dpts_list,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
            )

        return rms, K, D, rvecs, tvecs
    
    def undistort(self, distorted_img, undistorted_size=None, R=np.eye(3), K=None):
        """Undistort an image using the fisheye model"""

        if K is None:
            K = self._K
        
        if undistorted_size is None:
            undistorted_size = distorted_img.shape[:2]
            
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self._K,
            self._D,
            R,
            K,
            undistorted_size,
            cv2.CV_16SC2            
        )
        
        undistorted_img = cv2.remap(
            distorted_img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        return undistorted_img
    
    def projectPoints(self, object_points, skew=0, rvec=None, tvec=None):
        """Projects points using fisheye model.
        """
        
        if object_points.ndim == 2:
            object_points = np.expand_dims(object_points, 0)
        
        if rvec is None:
            rvec = np.zeros(3).reshape(1, 1, 3)
        else:
            rvec = np.array(rvec).reshape(1, 1, 3)
            
        if tvec is None:
            tvec = np.zeros(3).reshape(1, 1, 3)
        else:
            tvec = np.array(tvec).reshape(1, 1, 3)

        image_points, jacobian = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec,
            self._K,
            self._D,
            alpha=skew
        )
        
        return np.squeeze(image_points)
    
    def undistortPoints(self, distorted, R=np.eye(3), K=None):
        """Undistorts 2D points using fisheye model.
            """

        if distorted.ndim == 2:
            distorted = np.expand_dims(distorted, 0)
        if K is None:
            K = self._K
                    
        undistorted = cv2.fisheye.undistortPoints(
            distorted.astype(np.float32),
            self._K,
            self._D,
            R=R,
            P=K
        )
        
        return np.squeeze(undistorted)
    
    def save(self, filename):
        """Save the fisheye model."""
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load a previously saved fisheye model.
        Note: this is a classmethod.
        """
        
        with open(filename, 'rb') as f:
            tmp_obj = pickle.load(f)
        
        obj = FishEye(nx=tmp_obj._nx, ny=tmp_obj._ny)
        obj.__dict__.update(tmp_obj.__dict__)
        
        return obj
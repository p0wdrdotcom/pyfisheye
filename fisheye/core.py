from __future__ import print_function
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import os


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
    
    def calibrate(
        self,
        img_paths,
        max_iter=30,
        eps=0.001,
        show_imgs=False
        ):
        """"""
        
        chessboard_model = np.zeros((1, self._nx*self._ny, 3), np.float32)
        chessboard_model[0, :, :2] = np.mgrid[0:self._nx, 0:self._ny].T.reshape(-1, 2)
        
        #
        # Arrays to store the chessboard points and image points from all the images.
        #
        chess_3Dpts_list = []
        chess_2Dpts_list = []
        
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
        
        if show_imgs:
            cv2.namedWindow('success img', cv2.WINDOW_NORMAL)
            cv2.namedWindow('fail img', cv2.WINDOW_NORMAL)
            
        for fname in img_paths:
            
            if self._verbose:
                print('Processing img: %s...' % os.path.split(fname)[1], end="")
            
            #
            # Load the image.
            #
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
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
                    
                chess_3Dpts_list.append(chessboard_model)
        
                cb_2d_pts_subpix = cv2.cornerSubPix(
                    gray,
                    cb_2D_pts,
                    (11, 11),
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
                    cv2.imshow('success img', img)
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
            cv2.destroyAllWindows()
        
        flag = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(chess_3Dpts_list))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(chess_3Dpts_list))]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        ret, mtx, dist, rvecs1, tvecs1 = \
            cv2.fisheye.calibrate(
                chess_3Dpts_list,
                chess_2Dpts_list,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                flag,
                (3, 12, 0)
            )
        
        return ret, mtx, dist, rvecs, tvecs
    
    def undistortImg(self, distorted_img, new_K, new_size):
        """"""

        undistort_img = np.zeros(new_size, dtype=distorted_img.dtype)
        cv2.fisheye.undistortImage(distorted_img, K, D, undistort_img, K, (800, 800))

        return undistort_img
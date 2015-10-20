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
        self._K = np.zeros((3, 3))
        self._D = np.zeros((4, 1))
        
    def calibrate(
        self,
        img_paths,
        update_model=True,
        max_iter=30,
        eps=1e-6,
        show_imgs=False
        ):
        """"""
        
        chessboard_model = np.zeros((1, self._nx*self._ny, 3), np.float32)
        chessboard_model[0, :, :2] = np.mgrid[0:self._nx, 0:self._ny].T.reshape(-1, 2)
        
        #
        # Arrays to store the chessboard image points from all the images.
        #
        chess_2Dpts_list = []
        
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        
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
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
            )

        return rms, K, D, rvecs, tvecs
    
    def undistort(self, distorted_img, undistorted_size=None, R=np.eye(3), K=None):
        """"""

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
    
    #void cv::fisheye::undistortImage(InputArray distorted, OutputArray undistorted,
                                     #InputArray K, InputArray D, InputArray Knew, const Size& new_size)
    #{
        #Size size = new_size.area() != 0 ? new_size : distorted.size();
    
        #cv::Mat map1, map2;
        #fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(), Knew, size, CV_16SC2, map1, map2 );
        #cv::remap(distorted, undistorted, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
    #}

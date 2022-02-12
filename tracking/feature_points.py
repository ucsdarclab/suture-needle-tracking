import numpy as np
import pandas as pd
import transforms3d.quaternions as quaternions

import pdb

#read DLC h5 file to csv: https://forum.image.sc/t/what-data-are-stored-in-h5-and-pickle/50478

class FeaturePoints(object): 
    def __init__(self, mode, fp_num, 
                 DLC_result_file=None, 
                 r_n=None, K=None, sim_var=None): 
        """
        param mode: the mode of getting feature points (read_files, simulation)
        param fp_num: the number of feature points
        param DLC_result_file: path to the csv file that contains DLC's labelling results (default: None)
        param r_n: the radius of the needle (in m, default: None)
        param K: the camera calibration matrix (default: None)
        param sim_var: the variance of the simulated pixel noise (default: None)
        """

        self.fp_num = fp_num

        if mode == 'read_files': 
            assert not DLC_result_file is None
            #(frame,16), (frame, tip xy likelihood, body 1~3 xy likelihood, tail xy likelihood)
            self.DLC_results = pd.read_csv(DLC_result_file, header=2).values
        elif mode == 'simulation': 
            assert not r_n is None and not K is None and not sim_var is None
            self.r_n, self.K, self.sim_var = r_n, K, sim_var

    def getFeaturePointsFromFile(self): 
        """
        Rearrange the labelled feature points from self.DLC_results
        return feature_points: all feature points read from the DLC's result file
        """

        feature_points = np.zeros([self.DLC_results.shape[0], 2, self.fp_num]) #(tail, tip, body1~3)
        feature_points[:,:,0] = self.DLC_results[:,[13,14]] #tail
        feature_points[:,:,1] = self.DLC_results[:,[1,2]] #tip
        feature_points[:,:,2] = self.DLC_results[:,[4,5]] #body 1
        feature_points[:,:,3] = self.DLC_results[:,[7,8]] #body 2
        feature_points[:,:,4] = self.DLC_results[:,[10,11]] #body 3

        return feature_points

    def getSimulatedFeaturePoints(self, pose_cam_needle): 
        """
        Get the simulated feature points in a simulator
        param pose_cam_needle: the pose of the needle in the camera frame 
            (the needle should be defined by the unique needle frame, quat: xyzw)
        """

        #needle points in the needle frame
        thetas = np.linspace(np.pi/2, 3*np.pi/2, self.fp_num)
        needle_pts = np.zeros([4,thetas.shape[0]])
        needle_pts[0] = self.r_n * np.cos(thetas)
        needle_pts[1] = self.r_n * np.sin(thetas)
        needle_pts[3] = 1.

        #transformation matrix from the camera frame to the needle frame
        H_cam_needle = np.eye(4)
        quat_wxyz = np.zeros(4)
        quat_wxyz[0] = pose_cam_needle[-1]
        quat_wxyz[1:] = pose_cam_needle[-4:-1]
        H_cam_needle[:3,:3] = quaternions.quat2mat(quat_wxyz)
        H_cam_needle[:3,3] = pose_cam_needle[:3]

        #needle points in the camera frame
        needle_pts = np.matmul(H_cam_needle, needle_pts)[:3] #(3,?)

        #pixels of feature points
        needle_pxs = np.matmul(self.K, needle_pts/needle_pts[2])[:2] #(2,?)
        noise = np.random.normal(0., np.sqrt(self.sim_var), size=needle_pxs.shape)
        needle_pxs += noise

        #reorder the feature points (tail, tip, body)
        new_idx = np.zeros(needle_pxs.shape[1], dtype=int)
        new_idx[:2] = np.array([-1,0])
        new_idx[2:] = np.arange(1, self.fp_num-1, 1)
        feature_points = needle_pxs[:,new_idx]

        return feature_points

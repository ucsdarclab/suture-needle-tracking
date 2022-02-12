import os
import cv2
import torch
import numpy as np
import pandas as pd
import transforms3d.quaternions as quaternions

from tracking.feature_points import FeaturePoints as FeaturePoints
from tracking.ellipse_fitting import EllipseFitting as EllipseFitting
from tracking.needle_tracking import NeedleTracking as NeedleTracking
from tracking.needle_reconstruction import NeedleReconstruction as NeedleReconstruction

class RealTimeTracking(object): 
    def __init__(self, K, r_n, ref_pts_target, fp_num, 
                 W, V, pose_u, pose_var, particle_num=1000): 

        #pos of the feature pt in the target frame 
        self.ref_pts_target = ref_pts_target.copy()
        self.fp_num = fp_num

        self.particle_num = particle_num

        self.e_fitting = EllipseFitting()
        self.n_recon = NeedleReconstruction(K, r_n)
        self.n_track = NeedleTracking(K, r_n, W, V, self.ref_pts_target, self.fp_num)

        self.pose_u = pose_u
        self.pose_var = pose_var

        self.particles = None
        self.alphas = None

    def runTracking(self, feature_points, pose=None, action=None, img=None): 
        #dictionary of new observation features
        new_ob = {'f_pts': feature_points}

        #only fit the ellipse when self.pose_u is not initialized
        if self.pose_u is None and pose is None: 
            #fit the ellipse on the image
            e_equ_coeff = self.e_fitting.fitEllipse(feature_points)
            #update the info for needle reconstruction
            self.n_recon.getEllipseInfo(e_equ_coeff, feature_points)
            #reconstruct the pose of the target
            H_camera_target = self.n_recon.needlePoseReconstruction(self.ref_pts_target[:,0][:,None])
            if np.isnan(H_camera_target).any(): 
                return
            #transform H matrix to 7-dim pose (x, y, z, qx, qy, qz, qw)
            pos = H_camera_target[:3,3]
            quat = quaternions.mat2quat(H_camera_target[:3,:3]) #(qw, qx, qy, qz)
            pose = np.concatenate([pos, quat[1:], [quat[0]]]) #(7,), (qx, qy, qz, qw)

            print('Initialization...')
            self.particles, self.alphas = self.n_track.initParticles(pose, self.pose_var, \
                                                                     num=self.particle_num) #(7,n), (n,)
            self.pose_u = torch.from_numpy(self.n_track.particles2Pose(self.particles, self.alphas)) #(7,)
            H_camera_target = self.n_track.pose2HMatrix(self.pose_u[:,None]).detach().numpy()

        elif self.pose_u is None: 
            print('Initialization...')
            self.particles, self.alphas = self.n_track.initParticles(pose, self.pose_var, \
                                                                     num=self.particle_num) #(7,n), (n,)
            self.pose_u = torch.from_numpy(self.n_track.particles2Pose(self.particles, self.alphas)) #(7,)
            H_camera_target = self.n_track.pose2HMatrix(self.pose_u[:,None]).detach().numpy()

        else: 
            #update the info for needle reconstruction
            self.n_recon.getEllipseInfo(feature_points=feature_points)

            assert not action is None, 'The value of the action must be specified.'
            #resample the particles
            self.particles, self.alphas = self.n_track.resampleParticles(self.particles, self.alphas)
            #PF predict step
            self.pose_u, self.particles = self.n_track.PFPrediction(self.particles, self.alphas, action)
            #PF update step
            self.pose_u, self.alphas = self.n_track.PFUpdate(self.particles, self.alphas, new_ob)
            self.pose_u = torch.from_numpy(self.pose_u) #(7,)

            #transform 7-dim pose (x, y, z, qx, qy, qz, qw) to H matrix
            H_camera_target = self.n_track.pose2HMatrix(self.pose_u[:,None]).detach().numpy()

            #check if the needle is flipped
            real_z_dir = self.n_recon.zDirection()
            current_z_dir = np.sign(H_camera_target[2,2])
            if not real_z_dir == current_z_dir: 
                H_wtarget_rtarget = np.array([[-1, 0,  0, 0], 
                                              [ 0, 1,  0, 0], 
                                              [ 0, 0, -1, 0], 
                                              [ 0, 0,  0, 1]])
                H_camera_target = np.matmul(H_camera_target, H_wtarget_rtarget)
                #resample the particles
                corrected_pose = np.zeros(7)
                corrected_pose[:3] = H_camera_target[:3,3]
                corrected_quat = quaternions.mat2quat(H_camera_target[:3,:3]) #wxyz
                corrected_pose[-1] = corrected_quat[0]
                corrected_pose[3:-1] = corrected_quat[1:]
                self.particles, self.alphas = self.n_track.initParticles(corrected_pose, self.pose_var, \
                                                                         num=self.particle_num) #(7,n), (n,)
                self.pose_u = torch.from_numpy(self.n_track.particles2Pose(self.particles, self.alphas)) #(7,)
                H_camera_target = self.n_track.pose2HMatrix(self.pose_u[:,None]).detach().numpy()

        if not np.isnan(H_camera_target).any() and not img is None: 
            #reproject onto the image
            self.n_recon.projectEstNeedlePose(H_camera_target, img, show=True)

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, 
            choices=['read_files'], 
            help='the mode of getting the images')
    parser.add_argument('-Kf', '--K_file', type=str, required=True, 
            help='path to the csv file of the camera calibration matrix')
    parser.add_argument('-nr', '--needle_radius', type=float, required=True, 
            help='the radius of the needle (in m)')
    parser.add_argument('-rpf', '--reference_point_file', type=str, required=True, 
            help='path to the csv file for the 3D position of the reference points (homogeneous form)')
    parser.add_argument('-fpn', '--feature_point_num', type=int, default=5, 
            help='number of the feature points detected by DLC (currently must be 5)')
    parser.add_argument('-pv', '--pixel_var', type=float, default=5., 
            help='variance in the pixel space (unit: pixels)')
    parser.add_argument('-pn', '--particle_num', type=int, default=1000, 
            help='number of particles')
    parser.add_argument('-si', '--show_image', type=int, default=0, 
            help='whether to show images with tracking results or not')
    parser.add_argument('-if', '--img_folder', type=str, default=None, 
            help='folder to the presaved images')
    parser.add_argument('-af', '--action_file', type=str, default=None, 
            help='path to the csv file of the presaved actions')
    parser.add_argument('-Drf', '--DLC_result_file', type=str, default=None, 
            help='path to the csv file of the DLC results')
    args = parser.parse_args()

    torch.set_default_tensor_type(torch.DoubleTensor) #torch.float64

    #camera calibration matrix
    K = pd.read_csv(args.K_file, header=None).values #(3,3)

    #3D position of the reference points in the target frame
    #needle tail, needle tip
    ref_pts_target = pd.read_csv(args.reference_point_file, header=None).values #(4,2)
    assert ref_pts_target.shape[1] == 2, 'The number of reference points must be 2.'

    #number of feature points
    fp_num = args.feature_point_num
    assert fp_num == 5, 'Currently the number of feature points must be 5.'

    #motion model noise
    W = torch.diag(torch.tensor([1e-5, 1e-5, 1e-5, 1e-2, 1e-2, 1e-2])) #motion model noise, (6,)
    #observation model noise
    V = torch.diag(args.pixel_var*torch.ones(2*2+fp_num-2)) #(2*2+?-2,)

    #initialize the mean of needle pose
    pose_u = None
    #initialize the covariance of particles
    pose_var = torch.diag(torch.tensor([1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3])) #(6,6)

    rt_tracking = RealTimeTracking(K, args.needle_radius, ref_pts_target, fp_num, \
                                   W, V, pose_u, pose_var, args.particle_num)

    if args.mode == 'read_files': 
        assert not args.img_folder is None and not args.action_file is None, \
                'The folder of images and the file of actions must be specified.'
        assert not args.DLC_result_file is None, 'The file of DLC results must be specified.'

        #read images from args.img_folder and sort them
        img_files = os.listdir(args.img_folder)
        img_files = sorted(img_files, key=lambda x: x[5:9]) #framexxxx.jpg
        #read actions from args.action_file
        actions = pd.read_csv(args.action_file, header=0).values #pos, quat(xyzw)
        assert len(img_files)-1 == actions.shape[0], 'The number of actions should be 1 less than that of images.'

        #read the DLC results from args.DLC_result_file
        fp_generator = FeaturePoints(args.mode, fp_num, DLC_result_file=args.DLC_result_file)
        all_feature_points = fp_generator.getFeaturePointsFromFile()

        #the number to repeat tracking using the first frame, which helps converge to a better initialized pose
        delay_num = 50

        #run tracking sequencially
        for idx in range(len(img_files)+delay_num): 
            real_idx = max(0, idx-delay_num)

            if args.show_image: 
                img = cv2.imread(os.path.join(args.img_folder, img_files[real_idx]))
            else: 
                img = None

            if real_idx == 0: 
                action = np.array([0., 0., 0., 0., 0., 0., 1.])
            else: 
                action = actions[real_idx-1]

            rt_tracking.runTracking(all_feature_points[real_idx], action=action, img=img)

        if args.show_image: 
            cv2.destroyAllWindows()

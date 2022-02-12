'''
state: [pos, quat]
'''

import cv2
import torch
import itertools
import numpy as np
from scipy.stats import multivariate_normal

class NeedleTracking(object): 
    #W: noise covariance for the motion model
    #V: noise covariance for the observation model
    #ref_pts_target: pos of feature pts in the target frame, torch tensor, (4,n), homogeneous form
    #ref_pts_num: number of the feature pts used (different from the size of ref_pts_target!!!)
    def __init__(self, K, r_n, W, V, ref_pts_target, ref_pts_num): 
        self.K = K.copy()
        self.r_n = r_n

        self.W = W #(6,6)
        self.V = V #(?+5,?+5)
        self.ref_pts_target = torch.from_numpy(ref_pts_target)
        self.ref_pts_num = ref_pts_num

    #pose: (7,n), (x, y, z, q_x, q_y, q_z, q_w)
    def pose2HMatrix(self, pose, squeeze=True): 
        num = pose.shape[1]
        H_camera_target = torch.zeros([4,4,num]) #(4,4,n)
        quat = pose[3:] #(4,n)

        E = torch.zeros([3,4,num]) #(3,4,n)
        E[:,0] = -quat[:3] #(3,n)
        #q_x
        E[1,3] = -quat[0] #(n,)
        E[2,2] = quat[0] #(n,)
        #q_y
        E[0,3] = quat[1] #(n,)
        E[2,1] = -quat[1] #(n,)
        #q_z
        E[0,2] = -quat[2] #(n,)
        E[1,1] = quat[2] #(n,)
        #q_w
        E[0,1] = quat[-1] #(n,)
        E[1,2] = quat[-1] #(n,)
        E[2,3] = quat[-1] #(n,)

        G = torch.zeros([3,4,num]) #(3,4,n)
        G[:,0] = -quat[:3] #(3,n)
        #q_x
        G[1,3] = quat[0] #(n,)
        G[2,2] = -quat[0] #(n,)
        #q_y
        G[0,3] = -quat[1] #(n,)
        G[2,1] = quat[1] #(n,)
        #q_z
        G[0,2] = quat[2] #(n,)
        G[1,1] = -quat[2] #(n,)
        #q_w
        G[0,1] = quat[-1] #(n,)
        G[1,2] = quat[-1] #(n,)
        G[2,3] = quat[-1] #(n,)

        H_camera_target[:3,:3] = torch.einsum('ijb,jkb->ikb', E, torch.transpose(G, 0, 1)) #(3,4,n)*(4,3,n) -> (3,3,n)
        H_camera_target[:3,3] = pose[:3] #(3,n)
        H_camera_target[3,3] = 1

        if squeeze: 
            return torch.squeeze(H_camera_target) #(4,4)
        return H_camera_target #(4,4,n)

    #pose: (7,n), (x,y,z,q_x,q_y,q_z,q_w)
    def _pose2RefPxs(self, pose, reshape=True, squeeze=True): 
        H_camera_target = self.pose2HMatrix(pose, squeeze=False) #(4,4,n)
        #reference points in the camera frame
        ref_pts_camera = torch.einsum('ijb,jk->ikb', H_camera_target, self.ref_pts_target) #(4,?,n)
        ref_pts_camera = (ref_pts_camera / ref_pts_camera[2])[:3] #(3,?,n)
        #project the reference points onto the image plane
        K = torch.tensor(self.K) #(3,3)
        ref_pxs = torch.einsum('ij,jkb->ikb', K, ref_pts_camera)[:2] #(2,?,n)

        if not reshape and squeeze: 
            return ref_pxs[:,:,0] #(2,?)
        elif reshape: 
            ref_pxs = torch.transpose(ref_pxs, 0, 1) #(?,2,n)
            ref_pxs = torch.reshape(ref_pxs, [2*ref_pxs.shape[0],ref_pxs.shape[2]]) #(2*?,n)
            if squeeze: 
                return torch.squeeze(ref_pxs) #(2*?,) / (2*?,n)
        return ref_pxs #(2*?,n) / (2,?,n)

    #pixels: (2,?,) / (2,?,n)
    def _ellipseData(self, pixels, squeeze=True): 
        if pixels.ndim == 2: 
            pixels = pixels[:,:,None] #(2,?,1)

        #ellipse equation: ax^2 + 2bxy + cy^2 + 2dx + 2ey + 1 = 0
        D = torch.zeros([pixels.shape[1],5,pixels.shape[2]]) #(?,5,n)
        D[:,0] = torch.pow(pixels[0], 2) #(?,n)
        D[:,1] = 2 * pixels[0] * pixels[1] #(?,n) .* (?,n)
        D[:,2] = torch.pow(pixels[1], 2) #(?,n)
        D[:,3:] = torch.transpose(2*pixels, 0, 1) #(?,2,n)

        if squeeze: 
            return D[:,:,0] #(?,5)
        return D #(?,5,n)

    #five_pts_needle: (4,5), homogeneous form
    #pose: (7,n), (x,y,z,q_x,q_y,q_z,q_w)
    def _pose2Ellipse(self, five_pts_needle, pose, squeeze=True): 
        H_camera_target = self.pose2HMatrix(pose, squeeze=False) #(4,4,n)
        #five points in the camera frame
        five_pts_camera = torch.einsum('ijb,jk->ikb', H_camera_target, five_pts_needle) #(4,5,n)
        five_pts_camera = (five_pts_camera / five_pts_camera[2])[:3] #(3,5,n)
        #project the five points onto the image plane
        K = torch.tensor(self.K) #(3,3)
        five_pxs = torch.einsum('ij,jkb->ikb', K, five_pts_camera) #(3,5,n)

        D = self._ellipseData(five_pxs[:2], squeeze=False) #(5,5,n)
        D = torch.transpose(torch.transpose(D, 0, 2), 1, 2) #(n,5,5)
        f = -torch.ones([D.shape[0],5,1]) #(n,5,1)

        e_equ_coeffs = torch.linalg.solve(D, f)[:,:,0] #(n,5)

        if squeeze: 
            return torch.squeeze(e_equ_coeffs.T) #(5,) / (5,n)
        return e_equ_coeffs.T #(5,n)

    def _getFivePtsNeedle(self): 
        five_angles_needle = torch.arange(0, 2*np.pi, 2*np.pi/5) #(5,)
        five_pts_needle = torch.zeros([4,5]) #homogeneous form
        five_pts_needle[0] = self.r_n * torch.cos(five_angles_needle)
        five_pts_needle[1] = self.r_n * torch.sin(five_angles_needle)
        five_pts_needle[3] = 1.

        return five_pts_needle #(4,5)

    #pose: (7,n), (x,y,z,q_x,q_y,q_z,q_w)
    #f_pxs: (2,?)
    def _pose2EllipseError2(self, pose, f_pxs, with_pixels=0): 
        five_pts_needle = self._getFivePtsNeedle() #(4,5)
        e_coeffs = self._pose2Ellipse(five_pts_needle, pose, squeeze=False) #(5,n)
        D = self._ellipseData(f_pxs) #(?,5)
        ellip_err = torch.matmul(D, e_coeffs) + 1. #(?,n)

        if with_pixels == 0: 
            return ellip_err #(?,n)
        else: 
            assert f_pxs.shape[1] >= with_pixels+1
            pixels = self._pose2RefPxs(pose, squeeze=False) #(2*?,n)
            return torch.cat([pixels[:2*with_pixels], ellip_err[with_pixels:]], dim=0) #(2*p+?-p,n)

    #pose, action: (7,?), (x,y,z,q_x,q_y,q_z,q_w)
    #noise: (6,?), (x,y,z,a_x,a_y,a_z)
    def _motionModel(self, pose, action, noise): 
        p_ndim, a_ndim, n_ndim = pose.ndim, action.ndim, noise.ndim
        if p_ndim == 1: 
            pose = pose[:,None] #(7,1)
        if a_ndim == 1: 
            action = action[:,None] #(7,1)
        if n_ndim == 1: 
            noise = noise[:,None] #(6,1)
        assert pose.shape[1] == action.shape[1] == noise.shape[1]

        pos_next = pose[:3,:] + action[:3,:] + noise[:3,:] #(3,?)

        quat_hat_w = action[-1,:]*pose[-1,:] - torch.sum(action[3:-1,:]*pose[3:-1,:], dim=0) #(?,)
        quat_hat_v = action[-1,:][None,:]*pose[3:-1,:] + pose[-1,:][None,:]*action[3:-1,:] + \
                     torch.cross(action[3:-1,:], pose[3:-1,:], dim=0) #(3,?)
        axangle = (2*torch.acos(quat_hat_w)/(torch.sin(torch.acos(quat_hat_w))+1e-12))[None,:] * quat_hat_v + \
                  noise[3:,:] #(3,?)
        angle = torch.linalg.norm(axangle, dim=0) #(?,)
        axis = axangle / (angle+1e-12)[None,:] #(3,?)
        quat_next = torch.zeros([4,pose.shape[1]]) #(4,?)
        quat_next[-1,:] = torch.cos(angle/2) #(?,)
        quat_next[:3,:] = torch.sin(angle/2)[None,:] * axis #(3,?)

        if p_ndim == 1 or n_ndim == 1: 
            return torch.cat([pos_next, quat_next], dim=0)[:,0] #(7,)
        else: 
            return torch.cat([pos_next, quat_next], dim=0) #(7,?)

    #pose: (7,) / (7,n), (x,y,z,q_x,q_y,q_z,q_w)
    #noise: (4+?-2,)
    #f_pxs: detected pixels of feature points, #(2,?)
    def _observationModel(self, pose, noise, f_pxs=None): 
        p_ndim = pose.ndim
        if p_ndim == 1: 
            pose = pose[:,None] #(7,1)
        noise = noise[:,None] #(4+?-2,1)

        #(2*2+?-2,) / (2*2+?-2,n)
        return torch.squeeze(self._pose2EllipseError2(pose, f_pxs, with_pixels=2) + noise)

    #pose: (7,n), (x,y,z,q_x,q_y,q_z,q_w)
    #input_V: (?,?)
    #f_pxs: (2,?)
    def _getDerivedV(self, pose, input_V, f_pxs, remove_num=0): 
        #ellipse coefficients of the estimated poses
        five_pts_needle = self._getFivePtsNeedle() #(4,5)
        e_coeffs = self._pose2Ellipse(five_pts_needle, pose, squeeze=False) #(5,n)

        #4*[(ax+by+d)^2 + (bx+cy+e)^2]
        M = torch.zeros([2,3,pose.shape[1]]) #(2,3,n)
        M[0,0] = e_coeffs[0] #a
        M[0,1] = M[1,0] = e_coeffs[1] #b
        M[1,1] = e_coeffs[2] #c
        M[0,2] = e_coeffs[3] #d
        M[1,2] = e_coeffs[4] #e

        f_pxs_h = torch.cat([f_pxs, torch.ones([1,f_pxs.shape[1]])], dim=0) #(3,?)
        new_V = torch.einsum('ijb,jk->ikb', M, f_pxs_h) #(2,?,n)

        if not remove_num == 0: 
            new_V = new_V[:,remove_num:] #(2,?-p,n)

        new_V = 4 * (new_V[0]**2 + new_V[1]**2).T #(n,?) / (n,?-p)
        new_V = torch.diag_embed(new_V, dim1=0, dim2=1) #(?,?,n) / (?-p,?-p,n)
        new_V = torch.einsum('ijb,jk->ikb', new_V, input_V) #(?,?,n) / (?-p,?-p,n)

        return new_V #(?,?,n) / (?-p,?-p,n)

    # ---------- EKF ---------- #

    #pose_u, action: (7,), (x,y,z,q_x,q_y,q_z,q_w)
    #pose_var: (7,7)
    def EKFPrediction(self, pose_u, pose_var, action): 
        #prediction noise
        p_noise = torch.zeros(6, requires_grad=True) #(6,)
        #jacobian of motionModel w.r.t pose
        F = torch.autograd.functional.jacobian(lambda x: self._motionModel(x, action, p_noise), pose_u) #(7,7)
        #jacobian of motionModel w.r.t noise
        Q = torch.autograd.functional.jacobian(lambda x: self._motionModel(pose_u, action, x), p_noise) #(7,6)

        #new mean and covariance
        pose_u = self._motionModel(pose_u, action, p_noise) #(7,)
        pose_var = torch.matmul(F, torch.matmul(pose_var, F.T)) + torch.matmul(Q, torch.matmul(self.W, Q.T)) #(7,7)

        return pose_u, pose_var

    #pose_u: (7,)
    #pose_var: (7,7)
    #new_ob: {'f_pts': (2,?)}
    def EKFUpdate(self, pose_u, pose_var, new_ob): 
        #update noise
        u_noise = torch.zeros(4+self.ref_pts_num-2, requires_grad=True) #(2*2+?-2,)
        f_pxs = torch.from_numpy(new_ob['f_pts']) #(2,?)

        #covariance of the observation noise
        V = self.V.detach().clone() #(2*2+?-2,2*2+?-2)
        V[4:,4:] = self._getDerivedV(pose_u[:,None], self.V[4:,4:], f_pxs=f_pxs, \
                                     remove_num=2)[:,:,0] #(?-2,?-2)

        #jacobian of observationModel w.r.t. pose
        #(4+?-2,7)
        H = torch.autograd.functional.jacobian(lambda x: self._observationModel(x, u_noise, f_pxs), pose_u)
        #jacobian of observationModel w.r.t noise
        #(4+?-2,4+?-2)
        R = torch.autograd.functional.jacobian(lambda x: self._observationModel(pose_u, x, f_pxs), u_noise)

        #compute Kalman gain
        C = torch.matmul(pose_var, H.T) #(7,4+?-2)
        #(4+?-2,4+?-2)
        S = torch.matmul(H, torch.matmul(pose_var, H.T)) + torch.matmul(R, torch.matmul(V, R.T))
        #(7,4+?-2)
        K = torch.matmul(C, torch.linalg.inv(S))

        #new mean and covariance
        feature_pts = torch.reshape(f_pxs.T, [2*self.ref_pts_num,]) #(2*?,)
        ellip_err = torch.zeros(self.ref_pts_num-2) #(?-2,)
        z = torch.cat([feature_pts[:4], ellip_err]) #(4+?-2,)

        #(4+?-2,)
        m = self._observationModel(pose_u, u_noise, f_pxs)
        pose_u = pose_u + torch.squeeze(torch.matmul(K, (z-m)[:,None])) #(7,)
        #constrain the norm of quat in pose_u
        pose_u[3:] /= torch.linalg.norm(pose_u[3:])
        pose_var = pose_var - torch.matmul(K, torch.matmul(S, K.T)) #(7,7)

        return pose_u, pose_var

    # ---------- Particle Filter ---------- #

    #quat: (4,) or (n,4), (q_x, q_y, q_z, q_w)
    def _quat2Axangle(self, quat): 
        if quat.ndim == 1: 
            quat = quat[None,:] #(1,4)

        angle = 2 * np.arccos(quat[:,-1])[:,None] #(n,1)
        axis = quat[:,:-1] / (np.sin(0.5*angle) + 1e-12) #(n,3)

        return np.squeeze(angle[:,None] * axis) #(3,) / (n,3)

    #axangle: (3,) or (n,3), (a_x, a_y, a_z)
    def _axangle2Quat(self, axangle): 
        if axangle.ndim == 1: 
            axangle = axangle[None,:] #(1,3)

        angle = np.linalg.norm(axangle, axis=1)[:,None] #(n,1)
        axis = axangle / angle #(n,3)
        q_w = np.cos(0.5 * angle) #(n,1)
        q_xyz = np.sin(0.5 * angle) * axis #(n,3)

        return np.squeeze(np.concatenate([q_xyz, q_w], axis=1)) #(4,) / (n,4)

    #pose: (7,), (x,y,z,q_x,q_y,q_z,q_w)
    #cov: (6,6), (x,y,z,a_x,a_y,a_z)
    def initParticles(self, pose, cov, num): 
        #quat in pose to axangle
        pose_6 = np.concatenate([pose[:3], np.zeros(3)]) #(6,)
        pose_6[3:] = self._quat2Axangle(pose[3:].copy())

        #generate particles
        particles = multivariate_normal.rvs(pose_6, cov, size=num) #(n,6)
        #calculate weights
        alphas = multivariate_normal.pdf(particles, mean=pose_6, cov=cov) #(n,)
        alphas /= np.sum(alphas) #(n,)

        #axangle in particles to quat
        particles_7 = np.concatenate([particles[:,:3], np.zeros([num,4])], axis=1) #(n,7)
        particles_7[:,3:] = self._axangle2Quat(particles[:,3:].copy())

        return particles_7.T, alphas #(7,n), (n,)

    #particles: (7,n); alphas: (n,)
    def particles2Pose(self, particles, alphas): 
        return np.sum(particles * alphas[None,:], axis=1) #(7,)

    #particles: (7,n); alphas: (n,); action: (7,)
    def PFPrediction(self, particles, alphas, action): 
        num = particles.shape[1]
        #motion noise
        motion_noise = multivariate_normal.rvs(np.zeros(self.W.shape[0]), self.W, size=num).T #(6,n)
        #apply motion model to each particle
        action_tile = np.tile(action[:,None], [1,num]) #(7,n)
        particles = self._motionModel(torch.from_numpy(particles), torch.from_numpy(action_tile), \
                                      motion_noise).detach().numpy() #(7,n)
        #expectation of the pose
        pose_u = self.particles2Pose(particles, alphas) #(7,)

        return pose_u, particles #(7,), (7,n)

    #particles: (7,n); alphas: (n,)
    #new_ob: {'f_pts': (2,?)}
    def PFUpdate(self, particles, alphas, new_ob): 
        #observation noise
        ob_noise = torch.zeros(4+self.ref_pts_num-2) #(4+?-2,)
        f_pxs = torch.from_numpy(new_ob['f_pts']) #(2,?)

        #apply observation model to each particle
        #(4+?-2,n)
        ob_means = self._observationModel(torch.from_numpy(particles), ob_noise, f_pxs).detach().numpy()

        #new mean and covariance
        feature_pts = np.reshape(new_ob['f_pts'].T, [2*self.ref_pts_num,]) #(2*?,)
        ellip_err = np.zeros(self.ref_pts_num-2) #(?-2,)
        z = np.concatenate([feature_pts[:4], ellip_err]) #(4+?-2,)

        #covariance of the observation noise
        V = torch.tile(self.V[:,:,None], [1,1,alphas.shape[0]]) #(4+?-2,4+?-2,n)
        #(?-2,?-2,n)
        V[4:,4:] = self._getDerivedV(torch.from_numpy(particles), self.V[4:,4:], f_pxs=f_pxs, \
                                     remove_num=2)
        V = np.moveaxis(V.detach().numpy(), -1, 0) #(n,4+?-2,4+?-2)
        #calculate inverse of V
        V = np.linalg.inv(V)
        V = np.moveaxis(V, 0, -1) #(4+?-2,4+?-2,n)

        #calculate pdf of z conditioned on each particle
        ob_diffs = z[:,None] - ob_means #(4+?-2,n)
        #(4+?-2,1,n)
        ob_pdfs = np.matmul(V, ob_diffs[:,None,:], axes=[(0,1),(0,1),(0,1)])
        ob_pdfs = np.matmul(ob_diffs[None,:,:], ob_pdfs, axes=[(0,1),(0,1),(0,1)]) #(1,1,n)
        ob_pdfs = np.exp(-0.5 * np.squeeze(ob_pdfs)) #(n,)
        #update weights
        alphas = alphas * ob_pdfs / np.sum(alphas * ob_pdfs) #(n,)

        #expectation of the pose
        pose_u = self.particles2Pose(particles, alphas) #(7,)

        return pose_u, alphas #(7,), (n,)

    #particles: (7,n); alphas: (n,)
    def resampleParticles(self, particles, alphas): 
        #check effective number of particles
        num_eff = (1. / np.sum(alphas ** 2)) / alphas.shape[0] #ratio, scalar
        #print(num_eff)
        if num_eff < 0.25: 
            new_particles = np.zeros_like(particles) #(7,n)
            num = alphas.shape[0]
            idx, cumul_prob = 0, alphas[0]
            for i in range(num): 
                walk = np.random.uniform(0., 1./num) + float(i)/num #scalar
                while walk > cumul_prob: 
                    idx += 1
                    cumul_prob += alphas[idx]
                new_particles[:,i] = particles[:,idx]

            return new_particles, np.ones_like(alphas)/num #(7,), (n,)
        else: 
            return particles, alphas

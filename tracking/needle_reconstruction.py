'''
state: [pos, quat]
'''

import cv2
import itertools
import numpy as np

class NeedleReconstruction(object): 
    def __init__(self, K, r_n): 
        self.K = K.copy() #camera calibration matrix
        self.r_n = r_n #radius of the needle

        #for estimating the z-direction of the needle
        body_thetas = np.array([135, 180, 225]) * np.pi / 180
        self.pts_needle_nbody = np.zeros([4,3])
        self.pts_needle_nbody[0] = r_n * np.cos(body_thetas)
        self.pts_needle_nbody[1] = r_n * np.sin(body_thetas)
        self.pts_needle_nbody[3] = 1.

    def getEllipseInfo(self, e_equ_coeffs=None, feature_points=None): 
        if not e_equ_coeffs is None: 
            #e_equ_coeffs: ax^2 + 2bxy + cy^2 + 2dx + 2ey + f = 0
            #should be: ax^2 + bxy + cy^2 + dx + ey + f = 0
            self.e_equ_coeffs = e_equ_coeffs.copy() #coefficients of the ellipse equation
            self.e_equ_coeffs[1] *= 2. #b
            self.e_equ_coeffs[3] *= 2. #d
            self.e_equ_coeffs[4] *= 2. #e

        if not feature_points is None: 
            self.feature_pts = feature_points.copy() #pixels of feature points

    def zDirection(self): 
        """
        Decide the direction of the needle's z axis (pointing toward or away from the image plane)
        return 1 or -1: 1 for pointing toward and -1 for pointing away
        """

        y_nimg = np.zeros(3)
        y_nimg[:2] = self.feature_pts[:,1] - self.feature_pts[:,0] #tip - tail
        y_nimg /= np.linalg.norm(y_nimg)
        x_nimg = np.cross(y_nimg, np.array([0.,0.,1.]))
        H_img_nimg = np.eye(4)
        H_img_nimg[:3,0] = x_nimg
        H_img_nimg[:3,1] = y_nimg
        H_img_nimg[:2,3] = 0.5 * (self.feature_pts[:,0] + self.feature_pts[:,1])
        body_pts_num = self.feature_pts.shape[1] - 2
        body_pxs_img = np.concatenate([self.feature_pts[:,2:], np.zeros([1,body_pts_num]), \
                                       np.ones([1,body_pts_num])], axis=0) #(4,?)
        body_pxs_nimg = np.matmul(np.linalg.inv(H_img_nimg), body_pxs_img) #(4,?)

        return -np.sign(np.mean(body_pxs_nimg[0])), H_img_nimg

    def estZDirection(self, H_camera_target, H_img_nimg): 
        pts_camera_nbody = np.matmul(H_camera_target, self.pts_needle_nbody)[:3] #(3,3)
        pxs_img_nbody = np.matmul(self.K, pts_camera_nbody/pts_camera_nbody[2])[:2] #(2,3)
        hpxs_img_nbody = np.zeros([4,3]) #homogeneous form
        hpxs_img_nbody[:2] = pxs_img_nbody
        hpxs_img_nbody[3] = 1.
        pxs_nimg_nbody = np.matmul(np.linalg.inv(H_img_nimg), hpxs_img_nbody)[:2] #(2,3)

        return -np.sign(np.mean(pxs_nimg_nbody[0]))

    def _targetPlaneEigenSpaces(self): 
        C = np.zeros([3,3])
        C[0,0] = self.e_equ_coeffs[0] #a
        C[1,1] = self.e_equ_coeffs[2] #c
        C[2,2] = self.e_equ_coeffs[5] #f
        C[0,1], C[1,0] = self.e_equ_coeffs[1]/2., self.e_equ_coeffs[1]/2. #b/2
        C[0,2], C[2,0] = self.e_equ_coeffs[3]/2., self.e_equ_coeffs[3]/2. #d/2
        C[1,2], C[2,1] = self.e_equ_coeffs[4]/2., self.e_equ_coeffs[4]/2. #e/2
        C_n = np.matmul(self.K.T, np.matmul(C, self.K))

        eig_vals, eig_vecs = np.linalg.eig(C_n) #(3,), (3,3)
        idx_combs = np.array(tuple(itertools.permutations(range(3)))) #(6,3)
        E_vals, R_1s = [], []
        for i in range(idx_combs.shape[0]): 
            e_vals = eig_vals[idx_combs[i]].copy() #(3,)
            e_vecs = eig_vecs[:,idx_combs[i]].copy() #(3,3)
            #check the sign of (lam2-lam1)/(lam3-lam2)
            if (e_vals[1]-e_vals[0])/(e_vals[2]-e_vals[1]) < 0: 
                continue
            #check the determinant
            if abs(np.linalg.det(e_vecs)-1.) > 1e-3: 
                continue
            E_vals.append(e_vals)
            R_1s.append(e_vecs)
        E_vals = np.array(E_vals)
        R_1s = np.array(R_1s)

        return E_vals, R_1s #(?,3), (?,3,3)

    #eig_vals: (3,), R_1: (3,3)
    def _targetPlaneOrientation(self, eig_vals, R_1): 
        thetas = np.zeros(2)
        thetas[0] = np.arctan(np.sqrt((eig_vals[1]-eig_vals[0]) / (eig_vals[2]-eig_vals[1])))
        thetas[1] = -thetas[0]

        R_2 = np.zeros([3,3,2])
        R_2[:,:,0] = np.array([[np.cos(thetas[0]), 0, np.sin(thetas[0])], 
                               [0, 1, 0], 
                               [-np.sin(thetas[0]), 0, np.cos(thetas[0])]])
        R_2[:,:,1] = np.array([[np.cos(thetas[1]), 0, np.sin(thetas[1])], 
                               [0, 1, 0], 
                               [-np.sin(thetas[1]), 0, np.cos(thetas[1])]])

        R_C1 = np.matmul(R_1, R_2[:,:,0]) #(3,3) * (3,3)
        R_C2 = np.matmul(R_1, R_2[:,:,1]) #(3,3) * (3,3)

        return R_C1, R_C2 #(3,3), (3,3)

    #eig_vals: (3,), R_1: (3,3)
    def _target3DLocation(self, eig_vals, R_1): 
        R_C1, R_C2 = self._targetPlaneOrientation(eig_vals, R_1)

        delta = self.r_n * np.sqrt(-(eig_vals[1]-eig_vals[0])*(eig_vals[2]-eig_vals[1]) / (eig_vals[0]*eig_vals[2]))
        d = self.r_n * np.sqrt(-eig_vals[1]**2 / (eig_vals[0]*eig_vals[2]))

        target_center = np.array([delta, 0, d])[:,None] #(3,1)

        Ts = np.zeros([3,2]) #2 possible vectors
        Ts[:,0] = np.squeeze(np.matmul(R_C1, target_center)) #(3,3) * (3,1)
        Ts[:,1] = np.squeeze(np.matmul(R_C2, target_center)) #(3,3) * (3,1)

        ns = np.concatenate([R_C1[:,2][:,None], R_C2[:,2][:,None]], axis=1) #(3,2)

        return Ts, ns, d #(3,2), (3,2), real value

    #eig_vals: (3,), R_1: (3,3)
    #ref_pts_target: (4,1)
    def _target3DPose(self, eig_vals, R_1, ref_pts_target): 
        Ts, ns, d = self._target3DLocation(eig_vals, R_1)
        assert Ts.shape[1] == ns.shape[1]

        z_dir, _ = self.zDirection()
        ns *= z_dir

        feature_pts = np.concatenate([self.feature_pts[:,0][:,None], [[1]]], axis=0) #(3,1)
        ref_pts_normalized = np.matmul(np.linalg.inv(self.K), feature_pts) #(3,3) * (3,1), z = 1

        Rs = np.zeros([3,3,2])
        for i in range(Ts.shape[1]): 
            scale = d / np.dot(np.squeeze(ref_pts_normalized), ns[:,i])
            y = (Ts[:,i][:,None] - scale * ref_pts_normalized) / np.linalg.norm(Ts[:,i][:,None] - scale * ref_pts_normalized) #(3,1)
            x = np.cross(np.squeeze(y), ns[:,i])[:,None] #(3,1)
            Rs[:,:,i] = np.concatenate([x, y, ns[:,i][:,None]], axis=1)

        Hs = np.zeros([4,4,2]) #2 possible homogeneous transforms
        Hs[:3,3,:] = Ts
        Hs[3,3,:] = 1
        Hs[:3,:3] = Rs

        #ref_pts in the camera frame
        ref_pts_camera = np.matmul(Hs, ref_pts_target[:,:,None], axes=[(0,1),(0,1),(0,1)]) #(4,1,2)

        #ref_pts projected on the image plane
        ref_pts_pixel = np.squeeze(ref_pts_camera[:3,:,:] / ref_pts_camera[2,:,:][None,:,:]) #(3,2)
        ref_pts_pixel = np.matmul(self.K, ref_pts_pixel) #(3,2), 2 possible vectors
        diff_norms = np.linalg.norm(ref_pts_pixel[:2] - self.feature_pts[:,0][:,None], axis=0) #(2,)
        min_idx = np.argmin(diff_norms)
        T = Ts[:,min_idx][:,None] #(3,1)
        R = Rs[:,:,min_idx] #(3,3)

        return T, R, diff_norms[min_idx]

    #helper function
    def getTipDir(self, H_camera_target, needle_pixels): 
        #pos of needle center, in the target frame
        needle_center_target = np.array([[self.r_n/2.,0,0,1], [-self.r_n/2.,0,0,1]]).T #(4,2)
        #pos of needle center, in the camera frame
        needle_center_camera = np.matmul(H_camera_target, needle_center_target) #(4,4) * (4,2)
        needle_center_camera = needle_center_camera[:3] / needle_center_camera[2][None,:] #(3,2) ./ (1,2)
        #project needle center onto the image plane
        needle_center_pixels = np.matmul(self.K, needle_center_camera)[:2] #(3,3) * (3,2)
        #compare distance from needle center to the closest needle pixel
        dis_left = np.min(np.linalg.norm(needle_pixels - needle_center_pixels[:,0][:,None], axis=0))
        dis_right = np.min(np.linalg.norm(needle_pixels - needle_center_pixels[:,1][:,None], axis=0))
        if dis_left <= dis_right: 
            return 'left'
        else: 
            return 'right'

    def needlePoseReconstruction(self, ref_pts_target): 
        eig_vals, R_1s = self._targetPlaneEigenSpaces()
        assert eig_vals.shape[0] == R_1s.shape[0]
        pos_target_needle, R_target_needle, min_pixel_diff = None, None, 1e6 #(3,1), (3,3), real value
        for i in range(eig_vals.shape[0]): 
            T, R, pixel_diff = self._target3DPose(eig_vals[i], R_1s[i], ref_pts_target)
            if T[2] > 0 and pixel_diff < min_pixel_diff: 
                pos_target_needle = T.copy()
                R_target_needle = R.copy()
                min_pixel_diff = pixel_diff

        H_camera_target = np.zeros([4,4])
        H_camera_target[:3,:3] = R_target_needle
        H_camera_target[:3,3] = np.squeeze(pos_target_needle)
        H_camera_target[3,3] = 1

        return H_camera_target

    #test function
    def plotFittedEllipse(self, img): 
        #plot fitted ellipse
        e_char = self.e_character.astype(int)
        cv2.ellipse(img, tuple(e_char[:2]), tuple(e_char[2:4]), e_char[4], startAngle=0, endAngle=360, color=(255,255,255), thickness=3)

        #cv2.imwrite('plot/results/needle_tracking/fitted_ellipse.png', img[:,:,[2,1,0]])
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #test function
    def projectEstNeedlePose(self, H_camera_target, img, show=True): 
        pt_num = 5000
        needle_xyzs = np.zeros([4,pt_num])
        thetas = np.linspace(np.pi/2, 3*np.pi/2, pt_num) #(5000,)
        needle_xyzs[0] = self.r_n * np.cos(thetas)
        needle_xyzs[1] = self.r_n * np.sin(thetas)
        needle_xyzs[3] = 1.
        needle_xyzs = np.matmul(H_camera_target, needle_xyzs)[:3] #(3,?)

        needle_pixels = np.matmul(self.K, needle_xyzs/needle_xyzs[2][None,:]) #(3,3) * ((3,?) ./ (1,?))
        needle_pixels = needle_pixels.astype(int)
        for i in range(needle_pixels.shape[1]): 
            cv2.circle(img, tuple(needle_pixels[:2,i]), 1, (0,255,0), thickness=1) #green

        feature_pt_xyz = np.array([0, -self.r_n, 0, 1])[:,None] #(4,1)
        feature_pt_xyz = np.matmul(H_camera_target, feature_pt_xyz)[:3] #(3,1)

        feature_pt_pixel = np.matmul(self.K, feature_pt_xyz/feature_pt_xyz[2]) #(3,3) * ((3,1) ./ (1,))
        feature_pt_pixel = feature_pt_pixel.astype(int)
        cv2.circle(img, tuple(feature_pt_pixel[:2,0]), 3, (0,0,255), thickness=3) #red

        circle_center_xyz = np.array([0.,0.,0.,1.])[:,None] #(4,1)
        circle_center_xyz[3] = 1.
        circle_center_xyz = np.matmul(H_camera_target, circle_center_xyz)[:3] #(3,1)

        circle_center_pixel = np.matmul(self.K, circle_center_xyz/circle_center_xyz[2]) #(3,3) * ((3,1 ./ (1,)))
        circle_center_pixel = circle_center_pixel.astype(int)
        cv2.circle(img, tuple(circle_center_pixel[:2,0]), 3, (255,0,0), thickness=3) #blue

        if show: 
            cv2.imshow('image', img)
            cv2.waitKey(1)

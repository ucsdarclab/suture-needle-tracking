import numpy as np
import pandas as pd

class EllipseFitting(object): 

    #pixels: (2,5)
    def _ellipseData(self, pixels): 
        #ellipse equation: ax^2 + 2bxy + cy^2 + 2dx + 2ey + 1 = 0
        D = np.zeros([pixels.shape[1],5]) #(5,5)
        D[:,0] = np.power(pixels[0], 2) #(5,)
        D[:,1] = 2 * pixels[0] * pixels[1] #(5,) .* (5,)
        D[:,2] = np.power(pixels[1], 2) #(5,)
        D[:,3:] = 2*pixels.T #(5,2)

        return D #(5,5)

    #needle_pxs: (2,?)
    def fitEllipse(self, needle_pxs): 
        D = self._ellipseData(needle_pxs) #(5,5)
        f = -1. * np.ones([5,1]) #(5,1)
        #ellipse equation: ax^2 + 2bxy + cy^2 + 2dx + 2ey + f = 0
        ellipse_equ_coeffs = np.linalg.solve(D, f)[:,0] #(5,)

        return np.concatenate([ellipse_equ_coeffs, [1.]]) #(6,)

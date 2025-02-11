import numpy as np
import math


  

class Vector3D(np.ndarray):
    def __new__(cls, array:np.ndarray):
        assert array.shape[-1] == 3, "Vector must have 3 components"
        #assert array.dtype == float, "Vector must have float components"
        obj = array.view(cls)
        return obj
    def _normalize(self):
        norm = np.linalg.norm(self, axis=-1, keepdims=True)
        if not np.allclose(norm, 0):
            self /= norm
        else:
            raise ValueError("at least one vector has zero norm")

class Quaternion4D(np.ndarray):
    def __new__(cls, array:np.ndarray):
        assert array.shape[-1] == 4, "Quaternion must have 4 components"
        obj = array.view(cls)
        return obj
    
    @classmethod
    def from_scaler_vector(cls, s:np.ndarray, v:Vector3D):
        #assert s.shape == v.shape[:-1], "scaler and vector must have the same shape"
        return cls(np.concatenate([s, v], axis=-1))
    
    def __eq__(self, other):
        return np.allclose(self, other)

    def _normalize(self):
        norm = np.linalg.norm(self, axis=-1, keepdims=True)
        if not np.allclose(norm, 0):
            self /= norm
        else:
            raise ValueError("at least one Quaternion has zero norm")
    def __mul__(self, other):
        if isinstance(other, Quaternion4D):

            w1, x1, y1, z1 = [self[...,i].view(np.ndarray) for i in range(4)]
            w2, x2, y2, z2 = [other[...,i].view(np.ndarray) for i in range(4)]
            return Quaternion4D(np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                                          w1*x2 + x1*w2 + y1*z2 - z1*y2,
                                          w1*y2 - x1*z2 + y1*w2 + z1*x2,
                                          w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1))
        return NotImplemented
    
    def conjugate(self):
        return Quaternion4D(np.array([self[...,0], -self[...,1], -self[...,2], -self[...,3]]).reshape(self.shape))
    
    def rotate(self, vector:Vector3D):
        p_vector = Quaternion4D(np.concatenate([np.zeros_like(vector[...,0:1]), vector], axis=-1))
        return (self * p_vector * self.conjugate())[...,1:]
    def to_matrix(self):
        w, x, y, z = [self[...,i].view(np.ndarray) for i in range(4)]
        return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                         [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                         [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    def to_euler(self):
        w, x, y, z = [self[...,i].view(np.ndarray) for i in range(4)]
        # Roll (rotation about x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (rotation about y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
        
        # Yaw (rotation about z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll.squeeze(), pitch.squeeze(), yaw.squeeze()
        


def Rotation_Euler(vector:Vector3D,roll, pitch, yaw):
    # Convert the Euler angles to a rotation matrix
    R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
    # Rotate the vector by the rotation matrix
    #return np.dot(R, vector)
    return np.dot(vector,R.T)

def create_3d_curve():
    # Create a 3D curve
    t = np.linspace(-2*np.pi, 2*np.pi, 900)
    x = 0.3*np.sin(10*t)
    y = 0.3*np.cos(10*t)
    z = t 
    return x, y, z

def create_3d_surface():
    # Create a 3D surface
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2*np.pi, 30)
    theta, phi = np.meshgrid(theta, phi)
    r = 0.5*np.abs(3*np.cos(theta)**2-1)
    z = r * np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    

    return x, y, z


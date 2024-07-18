import numpy as np
from utils import Quaternion, Vector, Vector3D, Quaternion4D, Rotation_Euler, create_3d_curve, create_3d_surface
# import for 3d plotting
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x,y,z = create_3d_curve()
    x1,y1,z1 = np.zeros_like(x),np.zeros_like(y),np.zeros_like(z)
    x2,y2,z2 = np.zeros_like(x),np.zeros_like(y),np.zeros_like(z)

    theta = np.pi/3
    u = Vector(-3, 1, -2)
    q = Quaternion.from_scaler_vector(np.cos(theta/2),np.sin(theta/2)*u)
    q._normalize()

    for i in range(len(x)):
        v = Vector(x[i],y[i],z[i])
        x1[i],y1[i],z1[i] = q.rotate(v)
        roll, pitch, yaw = q.to_euler()
        x2[i],y2[i],z2[i] = Rotation_Euler(v,roll=roll, pitch=pitch, yaw=yaw)

    #fig = plt.figure()
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    #ax = fig.add_subplot(111, projection='3d')
    ax1.plot(x, y, z, label='original', color='blue')
    ax1.plot(x1, y1, z1, label='rotated1', color='green',alpha=0.5)
    ax1.plot(x2, y2, z2, label='rotated2', color='red',alpha=0.5)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_zlim(-5, 5)
    #plt.show()

    
    x,y,z = create_3d_surface()
    x1,y1,z1 = np.zeros_like(x),np.zeros_like(y),np.zeros_like(z)
    x2,y2,z2= np.zeros_like(x),np.zeros_like(y),np.zeros_like(z)
    for i in range(len(x)):
        for j in range(len(x[i])):
            v = Vector(x[i,j],y[i,j],z[i,j])
            x1[i,j],y1[i,j],z1[i,j] = q.rotate(v)
            roll, pitch, yaw = q.to_euler()
            x2[i,j],y2[i,j],z2[i,j] = Rotation_Euler(v,roll=roll, pitch=pitch, yaw=yaw)

    ax2.plot_surface(x, y, z, cmap='viridis', alpha=0.2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)

    ax2.plot_surface(x1, y1, z1, cmap='winter', alpha=0.4)
    ax2.plot_surface(x2, y2, z2, cmap='autumn', alpha=0.6)

    plt.show()


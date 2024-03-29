# %% Load libraies and the given paralleltomo function
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from skimage.measure import block_reduce
from skimage.morphology import disk
from skimage.util import random_noise
import scipy.io

im = np.load("testImage.npy")
#%%
def paralleltomo(N, theta=None, p=None, d=None):
    """
    PARALLELTOMO Creates a 2D tomography system matrix using parallel beams
    
      [A,theta,p,d] = paralleltomo(N)
      [A,theta,p,d] = paralleltomo(N,theta)
      [A,theta,p,d] = paralleltomo(N,theta,p)
      [A,theta,p,d] = paralleltomo(N,theta,p,d)
    
    This function creates a 2D tomography test problem with an N-times-N
    domain, using p parallel rays for each angle in the vector theta.
    
    Input: 
      N           Scalar denoting the number of discretization intervals in 
                  each dimesion, such that the domain consists of N^2 cells.
      theta       Vector containing the angles in degrees. Default: theta = 
                  0:1:179.
      p           Number of parallel rays for each angle. Default: p =
                  round(sqrt(2)*N).
      d           Scalar denoting the distance from the first ray to the last.
                  Default: d = sqrt(2)*N.
    
    Output:
      A           Coefficient matrix with N^2 columns and nA*p rows, 
                  where nA is the number of angles, i.e., length(theta).
      theta       Vector containing the used angles in degrees.
      p           The number of used rays for each angle.
      d           The distance between the first and the last ray.
    
    See also: fanbeamtomo, seismictomo.

    Anders Nymark Christensen, 20180216, DTU Compute

    Revised from the matlab version by:    
    Jakob Sauer Jørgensen, Maria Saxild-Hansen and Per Christian Hansen,
    October 1, 201r, DTU Compute.

    Reference: A. C. Kak and M. Slaney, Principles of Computerized 
    Tomographic Imaging, SIAM, Philadelphia, 2001.
    """

    # Default value of the angles theta.
    if theta is None:
        theta = np.matrix(np.arange(0.,180.))

    # Default value of the number of rays.
    if p is None:
        p = int(round(np.sqrt(2)*N))

    # Default value of d.
    if d is None:
        d = np.sqrt(2)*N

    # Define the number of angles.
    nA = theta.shape[1]

    # The starting values both the x and the y coordinates. 
    x0 = np.matrix(np.linspace(-d/2,d/2,p)).T
    y0 = np.matrix(np.zeros([p,1]))

    # The intersection lines.
    x = np.matrix(np.arange(-N/2,N/2 + 1)).T
    y = np.copy(x)

    # Initialize vectors that contains the row numbers, the column numbers and
    # the values for creating the matrix A effiecently.
    rows = np.matrix(np.zeros([2*N*nA*p,1]))
    cols = np.copy(rows)
    vals = np.copy(rows)
    idxend = 0

    # Loop over the chosen angles.
    for i in range(0,nA):
                
        # All the starting points for the current angle.
        x0theta = np.cos(np.deg2rad(theta[0,i]))*x0-np.sin(np.deg2rad(theta[0,i]))*y0
        y0theta = np.sin(np.deg2rad(theta[0,i]))*x0+np.cos(np.deg2rad(theta[0,i]))*y0
        
        # The direction vector for all the rays corresponding to the current 
        # angle.
        a = -np.sin(np.deg2rad(theta[0,i]))
        b = np.cos(np.deg2rad(theta[0,i]))
        
        # Loop over the rays.
        for j in range(0,p):
            # Use the parametrisation of line to get the y-coordinates of
            # intersections with x = k, i.e. x constant.
            tx = (x - x0theta[j,0])/a
            yx = b*tx + y0theta[j,0]
            
            # Use the parametrisation of line to get the x-coordinates of
            # intersections with y = k, i.e. y constant.
            ty = (y - y0theta[j,0])/b
            xy = a*ty + x0theta[j,0]            
            
            # Collect the intersection times and coordinates. 
            t = np.vstack([tx, ty])
            xxy = np.vstack([x, xy])
            yxy = np.vstack([yx, y])
            
            # Sort the coordinates according to intersection time.
            I = np.argsort(t,0)
            xxy = xxy[I]
            yxy = yxy[I]        
            
            # Skip the points outside the box.
            I1 = np.logical_and(np.array(xxy) >= -N/2 , np.array(xxy) <= N/2)
            I2 = np.logical_and(np.array(yxy) >= -N/2 , np.array(yxy) <= N/2)
            I = np.squeeze(np.logical_and(I1,I2))
            #I = (xxy >= -N/2 & xxy <= N/2 & yxy >= -N/2 & yxy <= N/2)
            xxy = np.squeeze(xxy[I])
            yxy = np.squeeze(yxy[I])
            
            # Skip double points.
            I = np.logical_and(abs(np.diff(xxy)) <= 1e-10 , abs(np.diff(yxy)) <= 1e-10)
            if np.not_equal(I.size, 0):
                I = np.concatenate((I, np.matrix([False])), axis=1)
            xxy = xxy[~I]
            yxy = yxy[~I]
            #xxy = np.delete(xxy,I)
            #yxy = np.delete(yxy,I)
            
            # Calculate the length within cell and determines the number of
            # cells which is hit.
            d = np.sqrt(np.power(np.diff(xxy),2) + np.power(np.diff(yxy),2))
            numvals = d.shape[1]
            
            # Store the values inside the box.
            if numvals > 0:
                
                # If the ray is on the boundary of the box in the top or to the
                # right the ray does not by definition lie with in a valid cell.
                if not ((b == 0 and abs(y0theta[j,0] - N/2) < 1e-15) or (a == 0 and abs(x0theta[j,0] - N/2) < 1e-15)):
                    
                    # Calculates the midpoints of the line within the cells.
                    xm = 0.5*(xxy[0,0:-1]+xxy[0,1:]) + N/2
                    ym = 0.5*(yxy[0,0:-1]+yxy[0,1:]) + N/2
                    
                    # Translate the midpoint coordinates to index.
                    col = np.floor(xm)*N + (N - np.floor(ym)) - 1
                    
                    # Create the indices to store the values to vector for
                    # later creation of A matrix.
                    idxstart = idxend
                    idxend = idxstart + numvals
                    idx = np.arange(idxstart,idxend)
                    
                    # Store row numbers, column numbers and values. 
                    rows[idx,0] = i*p + j
                    cols[idx,0] = col[0,:]
                    vals[idx,0] = d  

    # Truncate excess zeros.
    rows = rows[0:idxend]
    cols = cols[0:idxend]
    vals = vals[0:idxend]
    
    # Create sparse matrix A from the stored values.
    A = csr_matrix((vals[:,0].astype(float), (np.squeeze(np.array(rows[:,0]).astype(int)), np.squeeze(np.array(cols[:,0]).astype(int)))), dtype=float, shape=(p*nA, N**2)).toarray()

    return [A, theta, p, d]
def findNoisyPixels(typ, b, Type):
    if typ == 'iron':
        points = [[14, 14], [14,15], [15,14], [15,15]]
        final_min = 100000
        tmp = 0
        final_max = 0
        for j in range(1000):
            b_noised = addnoise(b, Type)
            #X_noised = A @ b_noised
            X_noised = np.linalg.lstsq(A, b_noised, rcond=None)[0]
            X = X_noised.reshape(N,N)
            p = np.empty((4,1))
            for i in range(4):
                p[i] = X[points[i][0], points[i][1]]
            if min(p) < final_min:
                final_min = min(p)
            if max(p) > final_max:
                final_max = max(p)
            if j % 10 == 0:
                print(f'Nået {(j/1000) * 100} % af vejen' )
        return final_min, final_max

    if typ == 'bismuth':
        points = [[16, 29], [16,30], [17,29], [17,30]]
    points_final = np.empty((1000,1))
    for j in range(1000):
        b_noised = addnoise(b, Type)
        #X_noised = A @ b_noised
        X_noised = np.linalg.lstsq(A, b_noised, rcond=None)[0]
        X = X_noised.reshape(N,N)
        p = np.empty((4,1))
        for i in range(4):
            p[i] = X[points[i][0], points[i][1]]
            
        points_final[j] = min(p)
        if j % 10 == 0:
            print(f'Nået {(j/1000) * 100} % af vejen' )
    return min(points_final)
def addnoise(b, Type):
    if Type == 1:
        noisemap = np.multiply(np.ones(b.shape) * (np.mean(b)), b > 0)
        b_noised = b + (np.random.poisson(noisemap) / 50)
    else:
        b_noised = b + random_noise(b, mode = 'poisson')
    return b_noised

# %% Downscale image
factor = 100
im_downscaled = block_reduce(im, factor)
plt.imshow(im_downscaled)
plt.colorbar()

# %% Find projections b
N = im_downscaled.shape[0]
x = im_downscaled.flatten()
[A, theta, p, d] = paralleltomo(N, np.matrix(np.arange(0.,180.,1.)))
b = A @ x
print(np.max(b))
# %% Reconstruct image
x_reconstructed = np.linalg.lstsq(A, b, rcond=None)[0]
im_reconstructed = x_reconstructed.reshape(N, N)
plt.imshow(im_reconstructed)
print(im_reconstructed)
# %% Reconstruct image with normal distributed noise
b_noised = addnoise(b, 1)
x_noised_reconstructed = np.linalg.lstsq(A, b_noised, rcond=None)[0]
im_noised_reconstructed = x_noised_reconstructed.reshape(N, N)
plt.imshow(im_noised_reconstructed)
plt.colorbar()

#%%
b_pois_noised = b + random_noise(b, mode = 'poisson')
x_pois_noised_reconstructed = np.linalg.lstsq(A, b_pois_noised, rcond=None)[0]
im_pois_noised_reconstructed = x_pois_noised_reconstructed.reshape(N, N)
plt.imshow(im_pois_noised_reconstructed)
plt.colorbar()

#%%
print(im_noised_reconstructed[17,30])

# %% Energies and Resolution
print(f'Resolution: N = (0.5 * 1000) / 2 = {(0.5 * 100) / (1 / 10)}')

# We use 60KeV x-ray sources
# 0.1844 (cm^2 / g) with 60 KeV from https://jwoodscience.springeropen.com/articles/10.1007/s10086-013-1381-z
mass_attenuation_coefficient_wood = 0.1844

# 1.205 (cm^2 / g) with 60 KeV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html
mass_attenuation_coefficient_iron = 1.205

# 5.233 (cm^2 / g) with 60 KeV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z83.html
mass_attenuation_coefficient_bismuth = 5.233

# We can see there use 100 KeV x-ray sources in the test data from
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html
#print("Attenuation values", np.unique(im_downscaled))
print(np.where(im_downscaled > 9))
print(im_downscaled[29,16])
# %% Downscale test
fig = plt.figure(dpi=200)
for i in range(1,32+1):
    plt.subplot(4, 8, i)
    plt.imshow(block_reduce(im, i*10))
    plt.axis('off')
    plt.title(str(i*10), )
plt.show()

# %% Degree test
im_downscaled = block_reduce(im, 100)
N = im_downscaled.shape[0]
x = im_downscaled.flatten()
print(max(x))
plt.imshow(im_downscaled)

fig = plt.figure(dpi=200)
for i in range(1,32+1):
    [A, theta, p, d] = paralleltomo(N, np.matrix(np.arange(0.,180.,i)))
    b = A @ x
    x_reconstructed = np.linalg.lstsq(A, b, rcond=None)[0]
    im_reconstructed = x_reconstructed.reshape(N, N)

    plt.subplot(4, 8, i)
    plt.imshow(im_reconstructed)
    plt.axis('off')
    plt.title(str(i))
plt.show()

# %%
mini, maxi = findNoisyPixels('iron', b, 2)

# Mindst fundne værdi på Bismuth efter poisson støj er 23.87473342

# %%
print(mini, maxi)

#%%
print(findNoisyPixels('bismuth',b, 2))
# %%
np.max(b)
# %%

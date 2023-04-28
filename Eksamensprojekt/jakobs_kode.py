# %% Load libraies and the given paralleltomo function
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from skimage.measure import block_reduce
from skimage.morphology import disk

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

    return (A, theta, p, d)

def reconstruct_im(im, degrees, downscale_factor=None, poisson_noise_lambda=None, poisson_noise_scale=None):
    if downscale_factor is not None:
        im = block_reduce(im, downscale_factor)

    N = im.shape[0]
    x = im.flatten()
    A, _, _, _ = paralleltomo(N, np.matrix(degrees))
    b = A @ x

    # Determine unique values in image & calculate the next power of two
    # vals = len(np.unique(im))
    # vals = 2 ** np.ceil(np.log2(vals))

    # Generating noise for each unique value in image.
    # out = rng.poisson(im * vals) / float(vals)

    if poisson_noise_lambda is not None:
        b_noise = poisson_noise_scale * np.random.poisson(poisson_noise_lambda, b.shape[0])
        b += b_noise
    
    x_recon = np.linalg.lstsq(A, b, rcond=None)[0]
    im_recon = x_recon.reshape(N, N)

    return im_recon

def simulate_im(r_log, r_bullets, mu_wood, mu_iron, mu_bismuth):
    im = disk(r_log) * mu_wood
    N = r_log*2 + 1

    for r_bullet in r_bullets:
        im_iron = disk(r_bullet) * mu_iron
        im_bismuth = disk(r_bullet) * mu_bismuth

        im_iron[im_iron == 0] = mu_wood
        im_bismuth[im_bismuth == 0] = mu_wood

        x_iron = np.random.randint(N)
        y_iron = np.random.randint(N)

        x_bismuth = np.random.randint(N)
        y_bismuth = np.random.randint(N)

        is_iron_inside_log = np.sqrt((x_iron - r_log)**2 + (y_iron - r_log)**2) < r_log - r_bullet
        is_bismuth_inside_log = np.sqrt((x_bismuth - r_log)**2 + (y_bismuth - r_log)**2) < r_log - r_bullet

        while not is_iron_inside_log:
            x_iron = np.random.randint(N)
            y_iron = np.random.randint(N)
            is_iron_inside_log = np.sqrt((x_iron - r_log)**2 + (y_iron - r_log)**2) < r_log - r_bullet

        while not is_bismuth_inside_log:
            x_bismuth = np.random.randint(N)
            y_bismuth = np.random.randint(N)
            is_bismuth_inside_log = np.sqrt((x_bismuth - r_log)**2 + (y_bismuth - r_log)**2) < r_log - r_bullet
        
        im[x_iron-r_bullet-1:x_iron+r_bullet, y_iron-r_bullet-1:y_iron+r_bullet] = im_iron
        im[x_bismuth-r_bullet-1:x_bismuth+r_bullet, y_bismuth-r_bullet-1:y_bismuth+r_bullet] = im_bismuth

    return im

def detect_bullets(im, mu_iron, mu_bismuth, delta_iron, delta_bismuth):
    im_mask_iron = mu_iron - delta_iron <= im <= mu_iron + delta_iron
    im_mask_bismuth = mu_bismuth - delta_bismuth <= im <= mu_bismuth + delta_bismuth

    n = im_mask_iron.shape[0]
    boxes_iron = []
    boxes_bismuth = []

    for x in range(n):
        for y in range(n):
            if im_mask_iron[x, y]:
                queue = []
                xs = []
                ys = []

                queue.push((x,y))
                xs.push(x)
                ys.push(y)
                im_mask_iron[x, y] = False

                while queue:
                    (x, y) = queue.pop()

                    if x + 1 < n and im_mask_iron[x+1, y]:
                        queue.push((x+1,y))
                        xs.push(x+1)
                        ys.push(y)
                        im_mask_iron[x+1, y] = False

                    if y + 1 < n and im_mask_iron[y+1, y]:
                        queue.push((x,y+1))
                        xs.push(x)
                        ys.push(y+1)
                        im_mask_iron[x, y+1] = False
                
                boxes_iron.push((min(xs), min(ys)), (max(xs), max(ys)), len(xs))
            
            if im_mask_bismuth[x, y]:
                queue = []
                xs = []
                ys = []

                queue.push((x,y))
                xs.push(x)
                ys.push(y)
                im_mask_bismuth[x, y] = False

                while queue:
                    (x, y) = queue.pop()

                    if x+1 < n and im_mask_bismuth[x+1, y]:
                        queue.push((x+1,y))
                        xs.push(x+1)
                        ys.push(y)
                        im_mask_bismuth[x+1, y] = False

                    if y+1 < n and im_mask_bismuth[y+1, y]:
                        queue.push((x,y+1))
                        xs.push(x)
                        ys.push(y+1)
                        im_mask_bismuth[x, y+1] = False
                
                boxes_bismuth.push((min(xs), min(ys)), (max(xs), max(ys)), len(xs))
    
    return (boxes_iron, boxes_bismuth)

# %%
# 0.1844 (cm^2 / g) with 60 KeV from https://jwoodscience.springeropen.com/articles/10.1007/s10086-013-1381-z
mu_wood = 0.1844

# 1.205 (cm^2 / g) with 60 KeV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html
mu_iron = 1.205

# 5.233 (cm^2 / g) with 60 KeV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z83.html
mu_bismuth = 5.233

# Radius for wood log and bullets
r_log = 2500
r_bullets = [10, 20, 30, 40, 50]

im = simulate_im(r_log, r_bullets, mu_wood, mu_iron, mu_bismuth)

degress = np.arange(0.,180.)
downscale_factor = 100
poisson_noise_lambda
im_recon = reconstruct_im(im, degress, downscale_factor, noise_mode)

# # %% Downscale image
# im = np.load("testImage.npy")
# im_downscaled = block_reduce(im, 100)
# plt.imshow(im_downscaled)

# # %% Find projections b
# N = im_downscaled.shape[0]
# x = im_downscaled.flatten()
# A, theta, p, d = paralleltomo(N, np.matrix(np.arange(0.,180.,1.)))
# b = A @ x

# # %% Reconstruct image
# x_reconstructed = np.linalg.lstsq(A, b, rcond=None)[0]
# im_reconstructed = x_reconstructed.reshape(N, N)
# plt.imshow(im_reconstructed)

# # %% Reconstruct image with normal distributed noise
# b_noise = np.random.normal(0, 1, b.shape[0])
# b_noised = b + b_noise
# x_noised_reconstructed = np.linalg.lstsq(A, b_noised, rcond=None)[0]
# im_noised_reconstructed = x_noised_reconstructed.reshape(N, N)
# plt.imshow(im_noised_reconstructed)

# # %% Energies and Resolution
# print(f'Resolution: N = (0.5 * 1000) / 2 = {(0.5 * 100) / (1 / 10)}')

# # We use 60KeV x-ray sources
# # 0.1844 (cm^2 / g) with 60 KeV from https://jwoodscience.springeropen.com/articles/10.1007/s10086-013-1381-z
# mu_wood = 0.1844

# # 1.205 (cm^2 / g) with 60 KeV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html
# mu_iron = 1.205

# # 5.233 (cm^2 / g) with 60 KeV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z83.html
# mu_bismuth = 5.233

# # We can see there use 100 KeV x-ray sources in the test data from
# # https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html
# print("Attenuation values", np.unique(im))

# # %% Downscale test
# fig = plt.figure(dpi=200)
# for i in range(1,32+1):
#     plt.subplot(4, 8, i)
#     plt.imshow(block_reduce(im, i*10))
#     plt.axis('off')
#     plt.title(str(i*10), )
# plt.show()

# # %% Degree test
# im_downscaled = block_reduce(im, 100)
# N = im_downscaled.shape[0]
# x = im_downscaled.flatten()
# plt.imshow(im_downscaled)

# fig = plt.figure(dpi=200)
# for i in range(1,32+1):
#     A, theta, p, d = paralleltomo(N, np.matrix(np.arange(0.,180.,i)))
#     b = A @ x
#     x_reconstructed = np.linalg.lstsq(A, b, rcond=None)[0]
#     im_reconstructed = x_reconstructed.reshape(N, N)

#     plt.subplot(4, 8, i)
#     plt.imshow(im_reconstructed)
#     plt.axis('off')
#     plt.title(str(i))
# plt.show()

# # %% Simulated test data
# [_, mu_wood, mu_iron, mu_bismuth] = np.unique(im)

# r_log = 2500
# r_bullets = [10, 20, 30, 40, 50]

# im_sim = disk(r_log) * mu_wood
# N_sim = r_log*2 + 1

# for r_bullet in r_bullets:
#     im_iron = disk(r_bullet) * mu_iron
#     im_bismuth = disk(r_bullet) * mu_bismuth

#     im_iron[im_iron == 0] = mu_wood
#     im_bismuth[im_bismuth == 0] = mu_wood

#     x_iron = np.random.randint(N_sim)
#     y_iron = np.random.randint(N_sim)

#     x_bismuth = np.random.randint(N_sim)
#     y_bismuth = np.random.randint(N_sim)

#     is_iron_inside_log = np.sqrt((x_iron - r_log)**2 + (y_iron - r_log)**2) < r_log - r_bullet
#     is_bismuth_inside_log = np.sqrt((x_bismuth - r_log)**2 + (y_bismuth - r_log)**2) < r_log - r_bullet

#     while not is_iron_inside_log:
#         x_iron = np.random.randint(N_sim)
#         y_iron = np.random.randint(N_sim)
#         is_iron_inside_log = np.sqrt((x_iron - r_log)**2 + (y_iron - r_log)**2) < r_log - r_bullet

#     while not is_bismuth_inside_log:
#         x_bismuth = np.random.randint(N_sim)
#         y_bismuth = np.random.randint(N_sim)
#         is_bismuth_inside_log = np.sqrt((x_bismuth - r_log)**2 + (y_bismuth - r_log)**2) < r_log - r_bullet
    
#     im_sim[x_iron-r_bullet-1:x_iron+r_bullet, y_iron-r_bullet-1:y_iron+r_bullet] = im_iron
#     im_sim[x_bismuth-r_bullet-1:x_bismuth+r_bullet, y_bismuth-r_bullet-1:y_bismuth+r_bullet] = im_bismuth

# plt.imshow(im_sim)

# # %%
# delta_iron = 0.01
# delta_bismuth = 0.01

# im_mask_iron = mu_iron - delta_iron <= im <= mu_iron + delta_iron
# im_mask_bismuth = mu_bismuth - delta_bismuth <= im <= mu_bismuth + delta_bismuth

# pos_iron = blob_log(im_mask_iron)
# pos_bismuth = blob_log(im_mask_bismuth)


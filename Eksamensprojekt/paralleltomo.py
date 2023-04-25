def paralleltomo(*args):
#PARALLELTOMO Creates a 2D tomography system matrix using parallel beams
#
#   [A,theta,p,d] = paralleltomo(N)
#   [A,theta,p,d] = paralleltomo(N,theta)
#   [A,theta,p,d] = paralleltomo(N,theta,p)
#   [A,theta,p,d] = paralleltomo(N,theta,p,d)
#
# This function creates a 2D tomography test problem with an N-times-N
# domain, using p parallel rays for each angle in the vector theta.
#
# Input: 
#   N           Scalar denoting the number of discretization intervals in 
#               each dimesion, such that the domain consists of N^2 cells.
#   theta       Vector containing the angles in degrees. Default: theta = 
#               0:1:179.
#   p           Number of parallel rays for each angle. Default: p =
#               round(sqrt(2)*N).
#   d           Scalar denoting the distance from the first ray to the last.
#               Default: d = sqrt(2)*N.
#
# Output:
#   A           Coefficient matrix with N^2 columns and nA*p rows, 
#               where nA is the number of angles, i.e., length(theta).
#   theta       Vector containing the used angles in degrees.
#   p           The number of used rays for each angle.
#   d           The distance between the first and the last ray.
# 
# See also: fanbeamtomo, seismictomo.

#Anders Nymark Christensen, 20180216, DTU Compute
#Revised from the matlab version by:
    
# Jakob Sauer Jørgensen, Maria Saxild-Hansen and Per Christian Hansen,
# October 1, 201r, DTU Compute.

# Reference: A. C. Kak and M. Slaney, Principles of Computerized 
# Tomographic Imaging, SIAM, Philadelphia, 2001.
    

    import numpy as np
    from scipy.sparse import csr_matrix
    
    N = args[0]

        
    # Default value of d.
    if len(args) < 4:
        d = np.sqrt(2)*N
    else:
        d = args[3]
    
    # Default value of the number of rays.
    if len(args) < 3:
        p = int(round(np.sqrt(2)*N))
    else:
        p = args[2]

    # Default value of the angles theta.
    if len(args) < 2:
        theta = np.matrix(np.arange(0.,180.))
    else:
        theta = args[1]


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
#            xxy = np.delete(xxy,I)
#            yxy = np.delete(yxy,I)
            
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
    A = csr_matrix((vals[:,0].astype(np.float), (np.squeeze(np.array(rows[:,0]).astype(int)), np.squeeze(np.array(cols[:,0]).astype(int)))), dtype=np.float, shape=(p*nA, N**2)).toarray()

    
    return [A,theta,p,d]

import numpy as np
N=8
theta = np.matrix([45.0000,   67.5000,   90.0000,  112.5000,  135.0000,  157.5000,  180.0000,  202.5000,  225.0000,  247.5000,  270.0000, 292.5000,  315.0000])
[A,theta,p,d] = paralleltomo(N,theta,11)

np.linalg.matrix_rank(A)

N=200
theta =np.matrix(np.linspace(0,179,179))
p = 250
[A,theta,p,d] = paralleltomo(N,theta,p)

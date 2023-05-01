# %% Load libraies and the given paralleltomo function
import numpy as np
import matplotlib.pyplot as plt
from numpy import inf
from itertools import islice
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix
from skimage.transform import rescale
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

                    if np.min(cols[:,0]) < 0:
                        garbage = 1 + 1

    # Truncate excess zeros.
    rows = rows[0:idxend]
    cols = cols[0:idxend]
    vals = vals[0:idxend]
    
    # Create sparse matrix A from the stored values.
    A = csr_matrix((vals[:,0].astype(float), (np.squeeze(np.array(rows[:,0]).astype(int)), np.squeeze(np.array(cols[:,0]).astype(int)))), dtype=float, shape=(p*nA, N**2)).toarray()

    return (A, theta, p, d)

def reconstruct_image(im, theta=None, p=None, d=None, noise_scale=None):
    N = im.shape[0]
    x = im.flatten()
    (A, _, _, _) = paralleltomo(N, np.matrix(theta), p, d)
    b = A @ x

    if noise_scale is not None:
        b += noise_scale * np.random.poisson(b)

    x_rec = np.linalg.lstsq(A, b, rcond=None)[0]
    im_rec = x_rec.reshape(N, N)

    return im_rec

def scale_image(im, scale):
    return rescale(im, scale, anti_aliasing=False)

def generate_image(r_log, r_bullets, mu_wood, mu_iron, mu_bis):
    im = disk(r_log) * mu_wood
    N = 2*r_log

    for r_bullet in r_bullets:
        im_mask_iron = disk(r_bullet, dtype=bool)
        im_mask_bis = disk(r_bullet, dtype=bool)

        x_iron = np.random.randint(N)
        y_iron = np.random.randint(N)

        x_bis = np.random.randint(N)
        y_bis = np.random.randint(N)

        is_iron_in_log = np.sqrt((x_iron - r_log)**2 + (y_iron - r_log)**2) < r_log - r_bullet
        is_bis_in_log = np.sqrt((x_bis - r_log)**2 + (y_bis - r_log)**2) < r_log - r_bullet

        while not is_iron_in_log:
            x_iron = np.random.randint(N)
            y_iron = np.random.randint(N)
            is_iron_in_log = np.sqrt((x_iron - r_log)**2 + (y_iron - r_log)**2) < r_log - r_bullet

        while not is_bis_in_log:
            x_bis = np.random.randint(N)
            y_bis = np.random.randint(N)
            is_bis_in_log = np.sqrt((x_bis - r_log)**2 + (y_bis - r_log)**2) < r_log - r_bullet

        im[x_iron-r_bullet-1:x_iron+r_bullet, y_iron-r_bullet-1:y_iron+r_bullet][im_mask_iron] = mu_iron
        im[x_bis-r_bullet-1:x_bis+r_bullet, y_bis-r_bullet-1:y_bis+r_bullet][im_mask_bis] = mu_bis

    return im[0:N, 0:N]

def detect_bullets(im, mu_iron, mu_bis, rel_error_iron, rel_error_bis):
    abs_error_iron = rel_error_iron * mu_iron
    abs_error_bis = rel_error_bis * mu_bis

    im_mask_iron = np.logical_and(
        mu_iron - abs_error_iron <= im, 
        im <= mu_iron + abs_error_iron
    )
    im_mask_bis = np.logical_and(
        mu_bis - abs_error_bis <= im, 
        im <= mu_bis + abs_error_bis
    )

    n = im_mask_iron.shape[0]
    boxes_iron = []
    boxes_bis = []

    for x in range(n):
        for y in range(n):
            if im_mask_iron[x, y]:
                queue = []
                xs = []
                ys = []

                queue.append((x,y))
                xs.append(x)
                ys.append(y)
                im_mask_iron[x, y] = False

                while len(queue) > 0:
                    (x, y) = queue.pop()

                    if x+1 < n and im_mask_iron[x+1, y]:
                        queue.append((x+1,y))
                        xs.append(x+1)
                        ys.append(y)
                        im_mask_iron[x+1, y] = False

                    if y+1 < n and im_mask_iron[x, y+1]:
                        queue.append((x,y+1))
                        xs.append(x)
                        ys.append(y+1)
                        im_mask_iron[x, y+1] = False
                    
                    if x-1 >= 0 and im_mask_iron[x-1, y]:
                        queue.append((x-1,y))
                        xs.append(x-1)
                        ys.append(y)
                        im_mask_iron[x-1, y] = False

                    if y-1 >= 0 and im_mask_iron[x, y-1]:
                        queue.append((x,y-1))
                        xs.append(x)
                        ys.append(y-1)
                        im_mask_iron[x, y-1] = False
                
                boxes_iron.append(((min(xs), min(ys)), (max(xs), max(ys)), len(xs)))
            
            if im_mask_bis[x, y]:
                queue = []
                xs = []
                ys = []

                queue.append((x,y))
                xs.append(x)
                ys.append(y)
                im_mask_bis[x, y] = False

                while queue:
                    (x, y) = queue.pop()

                    if x+1 < n and im_mask_bis[x+1, y]:
                        queue.append((x+1,y))
                        xs.append(x+1)
                        ys.append(y)
                        im_mask_bis[x+1, y] = False

                    if y+1 < n and im_mask_bis[x, y+1]:
                        queue.append((x,y+1))
                        xs.append(x)
                        ys.append(y+1)
                        im_mask_bis[x, y+1] = False

                    if x-1 >= 0 and im_mask_bis[x-1, y]:
                        queue.append((x-1,y))
                        xs.append(x-1)
                        ys.append(y)
                        im_mask_bis[x-1, y] = False

                    if y-1 >= 0 and im_mask_bis[x, y-1]:
                        queue.append((x,y-1))
                        xs.append(x)
                        ys.append(y-1)
                        im_mask_bis[x, y-1] = False
                
                boxes_bis.append(((min(xs), min(ys)), (max(xs), max(ys)), len(xs)))
    
    return (boxes_iron, boxes_bis)

# %% Find best condition number scaled down by a factor 5
conds = {}
N = 50

for degree in range(1, 5+1, 1):
    for p in range(25, 75+1, 5):
        for d in range(25, 75+1, 5):
            print()
            print(N, degree, p, d)
            try:
                theta = np.matrix(np.arange(0., 180., degree))
                (A, _, _, _) = paralleltomo(N, theta, p, d)
                cond = np.linalg.cond(A)
                print(cond)
                conds[(N, degree, p, d)] = cond
            except:
                continue

conds = dict(sorted(conds.items(), key=lambda item: item[1]))
print(conds)
exit()

# %% Generate image
mu_wood = 1.220
mu_iron = 449.45
mu_bis = 1265.54

radius_log = 25
radius_bullet = [0] * 10

im = generate_image(radius_log, radius_bullet, mu_wood, mu_iron, mu_bis)
plt.imshow(im)

# %% Detect bullets
rel_error_iron = 0.5
rel_error_bis = 0.5

(boxes_iron, boxes_bis) = detect_bullets(im, mu_iron, mu_bis, rel_error_iron, rel_error_bis)

fig, ax = plt.subplots()
ax.imshow(im)

for ((x1,y1), (x2,y2), n) in boxes_iron:
    ax.add_patch(Rectangle((y1-1, x1-1), y2-y1+2, x2-x1+2, linewidth=1, edgecolor='white', facecolor='none'))

for ((x1,y1), (x2,y2), n) in boxes_bis:
    ax.add_patch(Rectangle((y1-1, x1-1), y2-y1+2, x2-x1+2, linewidth=1, edgecolor='red', facecolor='none'))

print(len(boxes_iron), len(boxes_bis))
plt.show()

# %% Test bullet detection on condition numbers
conds = {(50, 5, 30, 45): 18.77584915914776, (50, 5, 30, 35): 42.07045152009125, (50, 4, 30, 45): 42.942219625941526, (50, 4, 30, 35): 54.22425914939154, (50, 5, 35, 40): 54.53691364434386, (50, 5, 35, 45): 56.287435289797656, (50, 5, 40, 45): 69.80424537324099, (50, 1, 65, 55): 77.7433152968374, (50, 2, 65, 55): 78.65475694974955, (50, 4, 35, 40): 81.8510056251662, (50, 4, 30, 40): 84.01366396255871, (50, 5, 30, 40): 87.57451942548656, (50, 1, 70, 55): 103.63864939362618, (50, 1, 65, 60): 105.25320861637331, (50, 4, 40, 45): 106.52134439195737, (50, 2, 70, 55): 107.66647662206131, (50, 2, 65, 60): 119.35471411116092, (50, 1, 55, 55): 119.7741093029527, (50, 1, 65, 65): 121.45884457083014, (50, 3, 70, 55): 135.935166301028, (50, 3, 65, 55): 136.28722607121637, (50, 3, 30, 45): 138.6056000572346, (50, 4, 35, 45): 156.05508035242525, (50, 1, 70, 45): 180.93699536879106, (50, 1, 65, 45): 182.13748131783655, (50, 2, 55, 55): 182.66137223384632, (50, 1, 60, 45): 188.06755299238972, (50, 5, 35, 35): 189.80415409657982, (50, 2, 70, 45): 190.2525104498996, (50, 5, 30, 30): 195.0435038359299, (50, 2, 65, 65): 195.90648474480668, (50, 1, 55, 45): 196.95929201905972, (50, 2, 65, 45): 203.98404807379984, (50, 4, 30, 30): 222.08951947207487, (50, 1, 45, 45): 223.44752964147992, (50, 2, 60, 45): 226.35589764924757, (50, 2, 55, 45): 241.25543859228185, (50, 1, 65, 70): 253.83758459962885, (50, 4, 70, 50): 270.45788482363463, (50, 2, 45, 45): 273.0604723805988, (50, 1, 55, 60): 278.0423454880443, (50, 3, 70, 45): 284.1693732148932, (50, 2, 65, 70): 296.3080863469962, (50, 4, 35, 35): 310.2561375242773, (50, 3, 30, 35): 318.25173145600627, (50, 2, 55, 60): 319.44097872215923, (50, 3, 65, 60): 324.3100756514814, (50, 4, 70, 55): 329.23686184115627, (50, 3, 65, 45): 344.9053321633464, (50, 5, 40, 40): 376.16100605087297, (50, 1, 60, 60): 381.7261116037931, (50, 1, 70, 70): 382.0978424165223, (50, 5, 45, 45): 386.0694132513654, (50, 3, 60, 45): 388.11956280016346, (50, 1, 55, 65): 395.97213915127315, (50, 2, 70, 70): 417.3719806649733, (50, 1, 50, 55): 419.66709206015884, (50, 1, 40, 55): 421.4702886588974, (50, 1, 60, 70): 425.0697393329348, (50, 3, 55, 45): 429.8398396020076, (50, 2, 60, 60): 432.890562897742, (50, 1, 60, 65): 434.41276367698515, (50, 2, 55, 65): 491.81479938564667, (50, 4, 65, 50): 501.6410856559049, (50, 3, 30, 30): 504.18387265198004, (50, 2, 70, 65): 504.4560318300392, (50, 1, 70, 65): 507.87460154053184, (50, 3, 30, 40): 521.8519104646596, (50, 2, 50, 55): 524.9335775187601, (50, 1, 50, 60): 526.7858472575787, (50, 1, 50, 65): 535.9135704377288, (50, 1, 40, 45): 552.5974621792068, (50, 2, 60, 65): 554.348877833663, (50, 3, 65, 65): 558.2490828138441, (50, 4, 40, 40): 565.8593633804867, (50, 2, 60, 70): 571.7398771285026, (50, 3, 55, 55): 583.7563398985474, (50, 2, 60, 55): 593.4765812237039, (50, 1, 60, 55): 619.185831751695, (50, 3, 70, 65): 624.710334961157, (50, 2, 40, 55): 647.1951403681032, (50, 1, 70, 60): 652.3951289033685, (50, 3, 35, 40): 656.5503740121976, (50, 2, 40, 45): 668.7880723780319, (50, 2, 50, 60): 674.6635117348997, (50, 2, 70, 60): 691.5285499970221, (50, 4, 45, 45): 720.3777737675973, (50, 3, 60, 55): 781.7954054904349, (50, 2, 50, 65): 799.5675894013008, (50, 4, 65, 55): 804.9806653835868, (50, 3, 35, 45): 811.6940268462932, (50, 1, 30, 45): 815.0208072904328, (50, 3, 70, 60): 821.388881737433, (50, 3, 60, 60): 913.8233903381822, (50, 1, 55, 40): 960.7480121234, (50, 3, 35, 35): 966.0103204293639, (50, 1, 50, 45): 967.7143030180007, (50, 1, 45, 40): 984.5878683241159, (50, 1, 50, 40): 990.9668605382359, (50, 1, 60, 40): 991.3812254650348, (50, 1, 40, 65): 996.6007751977944, (50, 2, 50, 45): 999.6459214575715, (50, 1, 65, 40): 1003.9422330244911, (50, 1, 70, 40): 1004.2044140142358, (50, 1, 40, 40): 1009.0873853715773, (50, 3, 70, 70): 1046.1822477418295, (50, 1, 35, 40): 1049.2905271090308, (50, 1, 55, 70): 1098.9339868723828, (50, 2, 70, 40): 1101.4940351420803, (50, 3, 65, 70): 1192.4124528874422, (50, 3, 50, 45): 1197.1864451793303, (50, 4, 70, 60): 1203.3164537559237, (50, 2, 65, 40): 1216.5967204324907, (50, 1, 35, 45): 1290.0029695102498, (50, 1, 60, 35): 1301.7613882816827, (50, 1, 55, 35): 1307.4287552263452, (50, 2, 60, 40): 1309.3847647721545, (50, 1, 70, 35): 1311.5652385341657, (50, 1, 45, 60): 1312.1165595725035, (50, 1, 50, 35): 1319.529153676943, (50, 1, 35, 35): 1329.1077560891472, (50, 3, 55, 60): 1332.7869627399932, (50, 2, 70, 35): 1336.778088750561, (50, 1, 65, 35): 1345.344849401954, (50, 1, 45, 35): 1347.6880130685606, (50, 2, 50, 40): 1365.3942035397472, (50, 2, 55, 40): 1400.6764284726432, (50, 2, 55, 70): 1408.9957025355368, (50, 1, 30, 40): 1416.7974900579075, (50, 3, 70, 40): 1424.812550394882, (50, 2, 45, 40): 1430.2186140555395, (50, 1, 30, 35): 1430.717819625467, (50, 1, 45, 30): 1430.765938664411, (50, 4, 70, 65): 1447.7623024808736, (50, 2, 65, 35): 1452.0260247582746, (50, 1, 50, 30): 1485.4940899732912, (50, 1, 35, 30): 1505.230626855574, (50, 2, 60, 35): 1542.3925761454964, (50, 3, 70, 35): 1551.4288500084892, (50, 1, 40, 30): 1568.7634763915758, (50, 1, 60, 30): 1580.9584857467792, (50, 1, 30, 30): 1597.4525323190226, (50, 3, 60, 65): 1603.691688994725, (50, 1, 55, 30): 1606.6115664869428, (50, 4, 65, 60): 1612.2842753742445, (50, 1, 70, 30): 1671.7725023966273, (50, 3, 50, 55): 1680.0746031245965, (50, 4, 70, 45): 1694.2875088780825, (50, 3, 55, 65): 1725.967163569271, (50, 1, 40, 35): 1727.5229498732854, (50, 2, 60, 30): 1736.1041697094379, (50, 2, 70, 30): 1746.642054053638, (50, 3, 60, 70): 1759.7212997836857, (50, 2, 55, 30): 1815.359098314222, (50, 2, 40, 60): 1851.597022751072, (50, 4, 60, 50): 1867.4104106495297, (50, 1, 65, 30): 1898.93176295116, (50, 3, 65, 40): 1910.0346592144683, (50, 2, 65, 30): 1951.859881699039, (50, 3, 70, 30): 1999.5231702783472, (50, 2, 55, 35): 2056.945190045459, (50, 3, 55, 40): 2073.234902713788, (50, 3, 65, 35): 2097.69969864551, (50, 3, 45, 45): 2108.049584779038, (50, 2, 45, 60): 2114.7246936211336, (50, 1, 30, 55): 2156.439756797172, (50, 2, 35, 40): 2288.023363581537, (50, 2, 50, 30): 2307.3044131268725, (50, 4, 70, 35): 2317.1044874656727, (50, 2, 50, 35): 2322.985803595223, (50, 4, 65, 45): 2323.442868852735, (50, 2, 40, 40): 2418.502675055667, (50, 1, 30, 60): 2502.032934228332, (50, 2, 35, 45): 2509.6981872070196, (50, 3, 60, 30): 2543.353020129508, (50, 3, 50, 40): 2552.6119861800926, (50, 4, 70, 40): 2580.150951969204, (50, 3, 60, 40): 2609.855974834298, (50, 3, 65, 30): 2617.3236252576285, (50, 2, 40, 35): 2625.86606101771, (50, 1, 45, 65): 2654.5150901598518, (50, 4, 60, 55): 2670.9687632449177, (50, 2, 45, 35): 2705.5042797431106, (50, 4, 70, 30): 2782.3349665167852, (50, 4, 60, 45): 3001.8453806564107, (50, 2, 45, 30): 3055.099286924531, (50, 3, 60, 35): 3103.6490483126154, (50, 3, 45, 40): 3149.7559095511533, (50, 3, 50, 60): 3235.5721129313597, (50, 4, 65, 40): 3809.1426571194306, (50, 3, 55, 30): 3812.222042569596, (50, 1, 45, 70): 3958.400507579337, (50, 4, 65, 35): 4185.1811163152715, (50, 1, 35, 55): 4291.564707092691, (50, 2, 35, 30): 4328.323220487162, (50, 1, 30, 65): 4900.857263298583, (50, 2, 45, 65): 5017.940953866203, (50, 3, 55, 35): 5257.732016613858, (50, 1, 40, 70): 5409.919239438506, (50, 4, 65, 30): 5511.920459226439, (50, 3, 50, 35): 5902.376182359968, (50, 2, 40, 30): 5981.010559187067, (50, 2, 35, 35): 5985.202424678073, (50, 2, 40, 65): 6073.525159626011, (50, 1, 35, 60): 6331.780140528943, (50, 3, 50, 30): 6388.546855481475, (50, 3, 45, 35): 6877.878034603416, (50, 1, 35, 65): 7427.919647366334, (50, 4, 60, 40): 7738.653286453026, (50, 4, 70, 70): 7760.311585198162, (50, 4, 60, 30): 9036.963849897074, (50, 2, 45, 70): 9056.230645905254, (50, 2, 30, 45): 9147.05413305657, (50, 2, 30, 40): 9393.662227908157, (50, 1, 30, 70): 9923.539171999959, (50, 4, 60, 35): 9945.642864683605, (50, 1, 35, 70): 11555.393046545043, (50, 2, 35, 55): 11753.566883544794, (50, 2, 30, 35): 12252.446943951601, (50, 3, 55, 70): 13509.166382504807, (50, 3, 40, 40): 16341.796646797935, (50, 4, 60, 60): 17189.60542561358, (50, 3, 50, 65): 18822.442032312658, (50, 4, 65, 65): 22419.356088333327, (50, 2, 30, 30): 24356.901050720724, (50, 3, 45, 30): 27186.557540054826, (50, 2, 35, 60): 36550.34773857994, (50, 2, 40, 70): 41463.179831216185, (50, 2, 35, 65): 76812.67788560796, (50, 3, 45, 60): 137812.49974092495, (50, 3, 40, 45): 289384.0885092681, (50, 2, 30, 55): 355169.5934681217, (50, 2, 35, 70): 6504656.443954722, (50, 4, 35, 50): 4363231362965365.5, (50, 5, 40, 35): 4785500515476339.0, (50, 5, 45, 40): 4917436408345463.0, (50, 5, 40, 30): 5014563515724756.0, (50, 5, 45, 35): 5346125461229035.0, (50, 5, 50, 45): 5898162249708283.0, (50, 4, 40, 35): 6076821376629233.0, (50, 5, 50, 40): 6203292503030179.0, (50, 5, 50, 35): 6578114975613927.0, (50, 4, 50, 50): 6709075761852019.0, (50, 4, 45, 35): 6760086036834209.0, (50, 4, 40, 50): 6784552366288780.0, (50, 4, 45, 40): 6906659024215494.0, (50, 5, 55, 45): 6907007630577476.0, (50, 4, 45, 50): 7667866200371249.0, (50, 5, 55, 40): 7842998902010562.0, (50, 5, 45, 30): 8217307653862203.0, (50, 5, 50, 30): 8512545290877755.0, (50, 4, 45, 30): 8531823289878544.0, (50, 4, 50, 45): 8700270357189074.0, (50, 5, 55, 35): 9211284841640654.0, (50, 5, 60, 45): 9998206372157678.0, (50, 4, 50, 40): 1.0175000005301278e+16, (50, 4, 50, 35): 1.120170036546952e+16, (50, 3, 35, 30): 1.135574260004786e+16, (50, 4, 40, 30): 1.1416587022395828e+16, (50, 5, 60, 40): 1.1671477034328556e+16, (50, 5, 55, 30): 1.1821908998654416e+16, (50, 4, 35, 30): 1.326258387766658e+16, (50, 4, 50, 30): 1.3284139718768542e+16, (50, 5, 60, 35): 1.4358890143453878e+16, (50, 5, 65, 45): 1.7297713753296402e+16, (50, 5, 60, 30): 1.7505636078418656e+16, (50, 4, 65, 70): 1.77496016555515e+16, (50, 3, 40, 35): 2.0062827772585916e+16, (50, 2, 30, 60): 2.166823287188904e+16, (50, 4, 30, 50): 2.409906025890987e+16, (50, 5, 65, 40): 2.425019508233549e+16, (50, 4, 60, 65): 2.6374304343014364e+16, (50, 5, 65, 35): 2.9264177441732836e+16, (50, 3, 45, 65): 3.075927168328504e+16, (50, 3, 40, 30): 3.24863761320222e+16, (50, 4, 55, 45): 3.5388988278737904e+16, (50, 5, 65, 30): 3.678095979353615e+16, (50, 4, 55, 50): 4.046866263764549e+16, (50, 2, 30, 65): 4.809528081448923e+16, (50, 4, 55, 40): 5.156453412976416e+16, (50, 3, 45, 70): 7.113209295521927e+16, (50, 4, 60, 70): 7.159376068290574e+16, (50, 4, 55, 35): 7.712314116450686e+16, (50, 4, 55, 30): 9.12030997695677e+16, (50, 5, 70, 45): 9.356868786781707e+16, (50, 5, 70, 55): 1.0235110636649429e+17, (50, 5, 35, 30): 1.1129785926882642e+17, (50, 2, 30, 70): 1.2209201110252379e+17, (50, 5, 70, 40): 1.3766834704556186e+17, (50, 5, 70, 35): 1.512224347634131e+17, (50, 5, 70, 30): 1.9889041178088557e+17, (50, 5, 70, 60): 3.9148210212551277e+17, (50, 5, 70, 65): 3.262957890426668e+32, (50, 5, 70, 70): 1.1130285751307404e+33, (50, 3, 30, 55): inf, (50, 3, 30, 60): inf, (50, 3, 30, 65): inf, (50, 3, 30, 70): inf, (50, 3, 35, 55): inf, (50, 3, 35, 60): inf, (50, 3, 35, 65): inf, (50, 3, 35, 70): inf, (50, 3, 40, 55): inf, (50, 3, 40, 60): inf, (50, 3, 40, 65): inf, (50, 3, 40, 70): inf, (50, 4, 30, 55): inf, (50, 4, 30, 60): inf, (50, 4, 30, 65): inf, (50, 4, 30, 70): inf, (50, 4, 35, 55): inf, (50, 4, 35, 60): inf, (50, 4, 35, 65): inf, (50, 4, 35, 70): inf, (50, 4, 40, 55): inf, (50, 4, 40, 60): inf, (50, 4, 40, 65): inf, (50, 4, 40, 70): inf, (50, 4, 45, 55): inf, (50, 4, 45, 60): inf, (50, 4, 45, 65): inf, (50, 4, 45, 70): inf, (50, 4, 50, 55): inf, (50, 4, 50, 60): inf, (50, 4, 50, 65): inf, (50, 4, 50, 70): inf, (50, 4, 55, 55): inf, (50, 4, 55, 60): inf, (50, 4, 55, 65): inf, (50, 4, 55, 70): inf, (50, 5, 30, 55): inf, (50, 5, 30, 60): inf, (50, 5, 30, 65): inf, (50, 5, 30, 70): inf, (50, 5, 35, 55): inf, (50, 5, 35, 60): inf, (50, 5, 35, 65): inf, (50, 5, 35, 70): inf, (50, 5, 40, 55): inf, (50, 5, 40, 60): inf, (50, 5, 40, 65): inf, (50, 5, 40, 70): inf, (50, 5, 45, 60): inf, (50, 5, 45, 65): inf, (50, 5, 45, 70): inf, (50, 5, 50, 55): inf, (50, 5, 50, 60): inf, (50, 5, 50, 65): inf, (50, 5, 55, 55): inf, (50, 5, 55, 60): inf, (50, 5, 55, 65): inf, (50, 5, 55, 70): inf, (50, 5, 60, 55): inf, (50, 5, 60, 60): inf, (50, 5, 60, 65): inf, (50, 5, 60, 70): inf, (50, 5, 65, 55): inf, (50, 5, 65, 60): inf, (50, 5, 65, 65): inf, (50, 5, 65, 70): inf}
start = 0

fig = plt.figure(dpi=1200)
for i, (k, v) in islice(enumerate(conds.items()), start, start+32):
    (_, degree, p, d) = k
    theta = np.matrix(np.arange(0., 180., degree))
    im_rec = reconstruct_image(im, theta, p, d)
    (boxes_iron, boxes_bis) = detect_bullets(im_rec, mu_iron, mu_bis, rel_error_iron, rel_error_bis)

    print(i, k, v, (len(boxes_iron), len(boxes_bis)))
    ax = plt.subplot(4, 8, i+1)
    ax.imshow(im_rec)
    ax.axis('off')
    ax.set_title(str(i))

    for ((x1,y1), (x2,y2), n) in boxes_iron:
        ax.add_patch(Rectangle((y1-1, x1-1), y2-y1+2, x2-x1+2, linewidth=1, edgecolor='white', facecolor='none'))

    for ((x1,y1), (x2,y2), n) in boxes_bis:
        ax.add_patch(Rectangle((y1-1, x1-1), y2-y1+2, x2-x1+2, linewidth=1, edgecolor='red', facecolor='none'))

plt.show()
# %%

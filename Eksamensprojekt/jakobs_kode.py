# %% Load libraies and the given paralleltomo function
import numpy as np
from numpy import inf
from itertools import islice
import matplotlib.pyplot as plt
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
    N = r_log*2 + 1

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

    return im

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

# %% Find best condition number
# conds = {}
# N = 51

# for degree in range(1, 15+1, 2):
#     for p in range(10, 70+1, 10):
#         for d in range(10, 70+1, 10):
#             print()
#             print(N, degree, p, d)
#             try:
#                 theta = np.matrix(np.arange(0., 180., degree))
#                 (A, _, _, _) = paralleltomo(N, theta, p, d)
#                 cond = np.linalg.cond(A)
#                 print(cond)
#                 conds[(N, degree, p, d)] = cond
#             except:
#                 continue

# print(conds)
# print(dict(sorted(conds.items(), key=lambda item: item[1])))

# key = min(conds, key=conds.get)
# print(key, conds[key])

# %% Generate image
# mu_wood = 1.220
# mu_iron = 449.45
# mu_bis = 1265.54

[_, mu_wood, mu_iron, mu_bis] = np.unique(np.load("testImage.npy"))

radius_log = 25
radius_bullet = [0] * 10

im = generate_image(radius_log, radius_bullet, mu_wood, mu_iron, mu_bis)

plt.imshow(im)


# # %% Reconstruct with optimal degree
# theta = np.arange(0.,180.,5.)
# p = N // 4
# d = N
# noise_scale = 0.1

# im_rec = reconstruct_image(im, theta, p, d, noise_scale)

# plt.imshow(im_rec)

# # %% Detect bullets
# rel_error_iron = 0.5
# rel_error_bis = 0.5

# (boxes_iron, boxes_bis) = detect_bullets(im_rec, mu_iron, mu_bis, rel_error_iron, rel_error_bis)

# fig, ax = plt.subplots()
# ax.imshow(im_rec)

# for ((x1,y1), (x2,y2), n) in boxes_iron:
#     height = x2 - x1
#     width = y2 - y1

#     rec = Rectangle((y1-1, x1-1), width+2, height+2, linewidth=1, edgecolor='white', facecolor='none')
#     ax.add_patch(rec)

# for ((x1,y1), (x2,y2), n) in boxes_bis:
#     height = x2 - x1
#     width = y2 - y1

#     rec = Rectangle((y1-1, x1-1), width+2, height+2, linewidth=1, edgecolor='red', facecolor='none')
#     ax.add_patch(rec)

# plt.show()

# # %% Test best degree on image
# im_sca = scale_image(im, 0.01)
# fig = plt.figure(dpi=200)

# for i in range(1,32+1):
#     theta = np.arange(0.,180.,i)
#     im_rec = reconstruct_image(im_sca, theta)

#     plt.subplot(4, 8, i)
#     plt.imshow(im_rec)
#     plt.axis('off')
#     plt.title(str(i))
# plt.show()


# %%
conds = {(51, 15, 10, 30): 2.6777408003931846, (51, 13, 10, 30): 2.6842825819293665, (51, 13, 10, 40): 2.754028124181845, (51, 15, 10, 40): 2.775250956649193, (51, 15, 10, 20): 2.8576311412081234, (51, 11, 10, 30): 3.0306718236601005, (51, 11, 10, 40): 3.125532970629845, (51, 13, 10, 20): 3.2820886789346466, (51, 13, 10, 50): 3.386212127811693, (51, 15, 10, 50): 3.407415998044359, (51, 9, 10, 30): 3.511785806827448, (51, 9, 10, 40): 3.6152797264543284, (51, 11, 10, 20): 3.724019591992377, (51, 11, 10, 50): 3.8100229829402985, (51, 7, 10, 30): 3.8680901177452673, (51, 9, 10, 20): 4.0674267966404765, (51, 7, 10, 40): 4.131870779735611, (51, 13, 20, 40): 4.419468563773107, (51, 15, 20, 40): 4.459630849435243, (51, 9, 10, 50): 4.544523658049322, (51, 15, 20, 30): 4.684161752111482, (51, 13, 20, 50): 4.868054839814016, (51, 11, 20, 40): 4.875658688765821, (51, 15, 20, 50): 4.888536289587723, (51, 7, 10, 20): 4.970683246693477, (51, 15, 10, 10): 5.252674470570559, (51, 11, 20, 50): 5.343448203488736, (51, 13, 20, 30): 5.382585919135002, (51, 7, 10, 50): 5.478171497502732, (51, 5, 10, 30): 5.560054465957942, (51, 5, 10, 40): 5.903695750982415, (51, 5, 10, 20): 5.9233787089742185, (51, 9, 20, 40): 6.19971231068162, (51, 11, 20, 30): 6.271330620946381, (51, 9, 20, 50): 6.556548095685783, (51, 13, 10, 10): 6.786149127643738, (51, 7, 20, 40): 7.043831212133979, (51, 9, 20, 30): 7.107033107817335, (51, 13, 30, 50): 7.886792129906575, (51, 11, 30, 50): 8.193271046923378, (51, 13, 30, 40): 8.299560301327904, (51, 7, 20, 50): 8.35742790997845, (51, 7, 20, 30): 8.715768067978605, (51, 9, 30, 50): 9.29924793128409, (51, 11, 10, 10): 9.65483307689129, (51, 11, 30, 40): 10.165614279846318, (51, 9, 10, 10): 10.172316080147665, (51, 5, 10, 50): 10.282726873631994, (51, 3, 10, 30): 10.361958734193895, (51, 15, 40, 50): 10.872046551413911, (51, 15, 30, 40): 10.882251392560931, (51, 5, 20, 40): 10.927828914706533, (51, 15, 30, 50): 12.293253888830508, (51, 7, 30, 50): 13.777008297875428, (51, 13, 40, 50): 14.036181741227416, (51, 15, 20, 20): 14.14534100470687, (51, 5, 20, 30): 14.220370165016954, (51, 7, 10, 10): 14.29378467029806, (51, 11, 40, 50): 14.408870473810985, (51, 3, 10, 40): 14.760744909606537, (51, 5, 20, 50): 14.858774075896713, (51, 3, 10, 20): 15.385301001247223, (51, 7, 30, 40): 15.657340277734699, (51, 9, 30, 40): 17.157389808672217, (51, 5, 10, 10): 17.959793650331996, (51, 13, 20, 20): 18.57078888543078, (51, 9, 40, 50): 19.50812789025192, (51, 15, 30, 30): 23.632246935858326, (51, 7, 40, 50): 27.64126628490654, (51, 3, 20, 30): 27.71495427598504, (51, 11, 20, 20): 27.96509966488089, (51, 9, 20, 20): 29.19834614497019, (51, 5, 30, 50): 29.87337301837966, (51, 3, 10, 50): 34.023991849972944, (51, 13, 30, 30): 36.220251212723795, (51, 3, 10, 10): 36.66045633891675, (51, 15, 40, 40): 39.81277106229664, (51, 7, 20, 20): 40.38324383918992, (51, 13, 40, 40): 51.44701053905244, (51, 5, 20, 20): 55.72195796745026, (51, 11, 30, 30): 56.81643227301077, (51, 9, 30, 30): 60.271352166257046, (51, 3, 20, 40): 60.935245495951825, (51, 3, 20, 50): 61.48171416127747, (51, 5, 30, 40): 74.14156412627273, (51, 7, 30, 30): 78.50174244288537, (51, 1, 70, 50): 82.30761088097758, (51, 11, 40, 40): 82.96958199082825, (51, 1, 60, 50): 86.09910649542553, (51, 5, 40, 50): 87.70246660838694, (51, 1, 70, 60): 92.29946870189436, (51, 15, 50, 50): 94.67682803061759, (51, 13, 50, 50): 95.41391576022299, (51, 9, 40, 40): 101.4763682878101, (51, 1, 50, 50): 104.56044689922494, (51, 1, 60, 60): 106.03807962232173, (51, 3, 70, 50): 116.08221581873575, (51, 1, 70, 70): 118.12547228298051, (51, 11, 50, 50): 124.5263458323249, (51, 5, 30, 30): 125.49384629955938, (51, 3, 20, 20): 126.74232678376235, (51, 7, 40, 40): 130.36466071445835, (51, 3, 70, 60): 140.9195285706006, (51, 3, 30, 50): 153.79256771018186, (51, 3, 60, 50): 195.93731418079471, (51, 5, 40, 40): 212.48519217944116, (51, 7, 50, 50): 259.4558841668589, (51, 9, 50, 50): 271.85378432243306, (51, 1, 50, 60): 301.2743112657811, (51, 3, 30, 30): 341.6559870379239, (51, 1, 60, 70): 350.22519751574566, (51, 3, 30, 40): 413.00758555636185, (51, 1, 40, 50): 518.0562626806819, (51, 3, 60, 60): 653.2998478219695, (51, 3, 70, 70): 666.7387800536129, (51, 5, 50, 50): 676.0720012178797, (51, 3, 50, 50): 958.7014004973511, (51, 1, 60, 40): 1069.0417172251016, (51, 1, 70, 40): 1080.4483060497569, (51, 1, 40, 40): 1085.6356546729132, (51, 1, 50, 40): 1111.8453626666278, (51, 3, 40, 50): 1229.2566741446585, (51, 3, 40, 40): 1521.2895186664389, (51, 1, 70, 30): 1587.549576484663, (51, 1, 50, 30): 1602.5542739983862, (51, 1, 40, 30): 1632.8916700347877, (51, 3, 60, 70): 1742.0003065993503, (51, 1, 60, 30): 1781.0758959870032, (51, 3, 70, 40): 1837.7831237840778, (51, 3, 70, 30): 1875.3785341810535, (51, 3, 50, 60): 1935.6508471170416, (51, 1, 30, 30): 1941.2952331751974, (51, 3, 60, 30): 3248.297665997922, (51, 1, 60, 20): 3267.112737857361, (51, 1, 70, 20): 3269.216416588779, (51, 3, 50, 40): 3283.5230845684946, (51, 1, 50, 20): 3558.5379922700063, (51, 1, 30, 20): 3666.401657232167, (51, 3, 60, 40): 4372.51029595668, (51, 3, 70, 20): 5092.855494021978, (51, 1, 30, 40): 5850.0520044145405, (51, 3, 60, 20): 7216.977114353264, (51, 1, 40, 20): 7552.371621660712, (51, 1, 20, 20): 11079.274288373937, (51, 3, 50, 30): 11800.863297700456, (51, 1, 60, 10): 13580.396649494269, (51, 1, 70, 10): 13617.890667413629, (51, 1, 50, 10): 13812.384465134595, (51, 1, 40, 10): 14618.365805928212, (51, 3, 50, 20): 21528.574706521988, (51, 1, 30, 10): 23099.048787992637, (51, 3, 70, 10): 29807.2702068242, (51, 3, 60, 10): 42504.516990048425, (51, 3, 50, 10): 125205.5626414551, (51, 7, 60, 50): 4388321795446784.0, (51, 7, 60, 40): 4582737473981698.0, (51, 7, 70, 50): 4837875825024284.0, (51, 5, 40, 30): 5092145100454155.0, (51, 5, 50, 40): 5623449830197714.0, (51, 7, 60, 30): 6299473301551085.0, (51, 7, 70, 40): 6356814514307180.0, (51, 5, 40, 20): 7707302355603776.0, (51, 5, 50, 30): 7975382078172341.0, (51, 5, 60, 50): 8981102694853845.0, (51, 3, 30, 20): 9527174905130730.0, (51, 7, 70, 30): 9648494292070640.0, (51, 5, 60, 40): 1.0077044775120652e+16, (51, 7, 70, 20): 1.091808295752985e+16, (51, 7, 60, 10): 1.327450558970612e+16, (51, 1, 20, 10): 1.3388077255979116e+16, (51, 7, 60, 20): 1.4816653814163168e+16, (51, 1, 20, 40): 1.5038993051172974e+16, (51, 5, 50, 20): 1.5042627012196786e+16, (51, 5, 60, 30): 1.5074898446826284e+16, (51, 7, 70, 10): 1.5234715623989366e+16, (51, 3, 50, 70): 1.595348907789381e+16, (51, 1, 10, 30): 1.6331290839585858e+16, (51, 1, 20, 50): 1.7216613124198872e+16, (51, 1, 10, 20): 1.780822011095547e+16, (51, 3, 40, 30): 1.867576989461155e+16, (51, 1, 10, 50): 1.9343098722951164e+16, (51, 1, 20, 30): 1.9534564418458264e+16, (51, 1, 10, 10): 1.994476626074745e+16, (51, 1, 10, 40): 2.221241012913195e+16, (51, 5, 40, 10): 2.497754821827679e+16, (51, 1, 20, 60): 2.5436777143289692e+16, (51, 5, 70, 50): 2.9477792386486332e+16, (51, 5, 60, 20): 3.014689767055822e+16, (51, 3, 30, 10): 3.2075433974071516e+16, (51, 3, 40, 20): 3.2366419573338972e+16, (51, 1, 20, 70): 3.5397822601778164e+16, (51, 5, 70, 40): 3.997649541451149e+16, (51, 11, 20, 10): 4.109582782312265e+16, (51, 7, 50, 40): 4.774012631952973e+16, (51, 1, 50, 70): 5.314801241511851e+16, (51, 1, 40, 70): 6.117671212917282e+16, (51, 5, 50, 10): 6.75644312510109e+16, (51, 5, 70, 30): 6.887279041366486e+16, (51, 11, 50, 30): 6.9278072617255224e+16, (51, 13, 50, 30): 8.749936860165635e+16, (51, 9, 40, 30): 8.970612404727296e+16, (51, 1, 30, 70): 9.054229755613688e+16, (51, 11, 30, 20): 9.516096617665918e+16, (51, 3, 40, 10): 9.763858565200608e+16, (51, 13, 60, 50): 1.0080940765367437e+17, (51, 13, 40, 20): 1.0118935558637698e+17, (51, 9, 50, 40): 1.1056739452441053e+17, (51, 15, 50, 40): 1.1159934987228027e+17, (51, 15, 40, 30): 1.1590488631211627e+17, (51, 15, 60, 60): 1.169404204324834e+17, (51, 11, 70, 50): 1.2363557536857757e+17, (51, 13, 70, 50): 1.2554734375720741e+17, (51, 15, 60, 50): 1.2755481555679254e+17, (51, 7, 40, 30): 1.2969602395802171e+17, (51, 13, 20, 10): 1.3636270959095714e+17, (51, 5, 60, 10): 1.425034244780803e+17, (51, 13, 30, 20): 1.4345014717999381e+17, (51, 11, 60, 50): 1.455948427410832e+17, (51, 5, 70, 20): 1.4694948041293027e+17, (51, 1, 40, 60): 1.4734835023857194e+17, (51, 15, 30, 20): 1.5378692923001037e+17, (51, 11, 40, 20): 1.7033772879107827e+17, (51, 13, 40, 30): 1.8675490608963914e+17, (51, 15, 60, 40): 1.875819279536333e+17, (51, 7, 50, 30): 1.9440928077664525e+17, (51, 15, 20, 10): 1.9578898445325942e+17, (51, 7, 50, 10): 2.017282017689791e+17, (51, 15, 30, 60): 2.1080004268832643e+17, (51, 13, 50, 20): 2.1305827942204656e+17, (51, 9, 40, 20): 2.1306128188377834e+17, (51, 11, 70, 30): 2.1914674112998794e+17, (51, 7, 20, 10): 2.2307661286455747e+17, (51, 7, 30, 20): 2.262039391415895e+17, (51, 11, 40, 30): 2.2826488944639318e+17, (51, 15, 20, 60): 2.3156582595788387e+17, (51, 15, 60, 30): 2.758871312118974e+17, (51, 13, 50, 40): 2.8133026371690323e+17, (51, 13, 70, 40): 3.093981593526814e+17, (51, 9, 60, 40): 3.206516000305085e+17, (51, 15, 40, 20): 3.3607690614448403e+17, (51, 11, 60, 40): 3.414327503803788e+17, (51, 15, 50, 60): 3.423806792118995e+17, (51, 7, 30, 10): 3.4519409143840454e+17, (51, 9, 50, 30): 3.469562465000842e+17, (51, 11, 50, 40): 3.7050550550074605e+17, (51, 13, 60, 40): 3.761162749584626e+17, (51, 9, 30, 20): 4.2245624229703443e+17, (51, 9, 30, 10): 4.4394121435800877e+17, (51, 5, 70, 10): 4.901113879607582e+17, (51, 11, 70, 40): 5.1266879307204595e+17, (51, 5, 30, 20): 5.430553227346428e+17, (51, 13, 50, 10): 5.4839181028602106e+17, (51, 15, 70, 60): 5.571753125435429e+17, (51, 13, 60, 30): 5.93880484337413e+17, (51, 11, 60, 30): 6.362787478548535e+17, (51, 1, 30, 60): 6.402552053539112e+17, (51, 7, 50, 20): 6.750658808315137e+17, (51, 15, 40, 60): 6.784558519501306e+17, (51, 11, 40, 10): 7.041547587634525e+17, (51, 11, 50, 20): 7.11570784835977e+17, (51, 15, 30, 10): 7.553844558959183e+17, (51, 13, 40, 10): 7.556463299257544e+17, (51, 9, 20, 10): 7.686785715723073e+17, (51, 13, 60, 20): 7.69071721160206e+17, (51, 5, 30, 10): 8.064878003651322e+17, (51, 15, 50, 30): 8.23522853507255e+17, (51, 13, 70, 30): 8.415958680081699e+17, (51, 9, 70, 50): 9.266112326528653e+17, (51, 9, 60, 30): 9.73285958491098e+17, (51, 11, 50, 10): 9.823719669049893e+17, (51, 1, 30, 50): 1.001263413455528e+18, (51, 7, 40, 10): 1.1092572892516027e+18, (51, 11, 70, 10): 1.1338372837529522e+18, (51, 15, 40, 10): 1.2635441464046943e+18, (51, 15, 50, 20): 1.2896953273621128e+18, (51, 9, 60, 20): 1.4637299140346967e+18, (51, 5, 20, 10): 1.5351814536552307e+18, (51, 11, 60, 10): 1.5505622267495967e+18, (51, 15, 70, 20): 1.6006396848996416e+18, (51, 15, 70, 40): 1.6340400430389233e+18, (51, 15, 70, 50): 1.8010587037318774e+18, (51, 3, 20, 10): 1.9211850152053658e+18, (51, 11, 70, 20): 2.061725981603676e+18, (51, 9, 50, 10): 2.1699473676568118e+18, (51, 15, 70, 30): 2.421150076774367e+18, (51, 13, 70, 10): 2.548182920477189e+18, (51, 9, 70, 40): 2.55287368604508e+18, (51, 9, 70, 20): 2.934923340354803e+18, (51, 15, 50, 10): 3.0022033434793795e+18, (51, 13, 70, 20): 3.039126454984176e+18, (51, 11, 60, 20): 3.170719507960512e+18, (51, 9, 50, 20): 3.28450054734429e+18, (51, 7, 40, 20): 3.4157926753663227e+18, (51, 9, 70, 30): 4.730710374407693e+18, (51, 9, 70, 10): 4.799390911476207e+18, (51, 15, 60, 20): 6.488514184818404e+18, (51, 13, 30, 10): 7.679671551596383e+18, (51, 13, 60, 10): 1.039991109533954e+19, (51, 15, 60, 10): 1.372613627892892e+19, (51, 9, 40, 10): 1.9843955863812354e+19, (51, 9, 60, 50): 4.67641421795903e+19, (51, 11, 30, 10): 8.095635940687541e+19, (51, 9, 60, 10): 9.353934271126202e+19, (51, 15, 70, 10): 9.624856960520584e+19, (51, 1, 10, 60): inf, (51, 1, 10, 70): inf, (51, 3, 10, 60): inf, (51, 3, 10, 70): inf, (51, 3, 20, 60): inf, (51, 3, 20, 70): inf, (51, 3, 30, 60): inf, (51, 3, 30, 70): inf, (51, 3, 40, 60): inf, (51, 3, 40, 70): inf, (51, 5, 10, 60): inf, (51, 5, 10, 70): inf, (51, 5, 20, 60): inf, (51, 5, 20, 70): inf, (51, 5, 30, 60): inf, (51, 5, 30, 70): inf, (51, 5, 40, 60): inf, (51, 5, 40, 70): inf, (51, 5, 50, 60): inf, (51, 5, 50, 70): inf, (51, 5, 60, 60): inf, (51, 5, 60, 70): inf, (51, 5, 70, 60): inf, (51, 5, 70, 70): inf, (51, 7, 10, 60): inf, (51, 7, 10, 70): inf, (51, 7, 20, 60): inf, (51, 7, 20, 70): inf, (51, 7, 30, 60): inf, (51, 7, 30, 70): inf, (51, 7, 40, 60): inf, (51, 7, 40, 70): inf, (51, 7, 50, 60): inf, (51, 7, 50, 70): inf, (51, 7, 60, 60): inf, (51, 7, 60, 70): inf, (51, 7, 70, 60): inf, (51, 7, 70, 70): inf, (51, 9, 10, 60): inf, (51, 9, 10, 70): inf, (51, 9, 20, 60): inf, (51, 9, 20, 70): inf, (51, 9, 30, 60): inf, (51, 9, 30, 70): inf, (51, 9, 40, 60): inf, (51, 9, 40, 70): inf, (51, 9, 50, 60): inf, (51, 9, 50, 70): inf, (51, 9, 60, 60): inf, (51, 9, 60, 70): inf, (51, 9, 70, 60): inf, (51, 9, 70, 70): inf, (51, 11, 10, 60): inf, (51, 11, 10, 70): inf, (51, 11, 20, 60): inf, (51, 11, 20, 70): inf, (51, 11, 30, 60): inf, (51, 11, 30, 70): inf, (51, 11, 40, 60): inf, (51, 11, 40, 70): inf, (51, 11, 50, 60): inf, (51, 11, 50, 70): inf, (51, 11, 60, 60): inf, (51, 11, 60, 70): inf, (51, 11, 70, 60): inf, (51, 11, 70, 70): inf, (51, 13, 10, 60): inf, (51, 13, 10, 70): inf, (51, 13, 20, 60): inf, (51, 13, 20, 70): inf, (51, 13, 30, 60): inf, (51, 13, 30, 70): inf, (51, 13, 40, 60): inf, (51, 13, 40, 70): inf, (51, 13, 50, 60): inf, (51, 13, 50, 70): inf, (51, 13, 60, 60): inf, (51, 13, 60, 70): inf, (51, 13, 70, 60): inf, (51, 13, 70, 70): inf, (51, 15, 10, 60): inf, (51, 15, 10, 70): inf, (51, 15, 20, 70): inf, (51, 15, 30, 70): inf, (51, 15, 40, 70): inf, (51, 15, 50, 70): inf, (51, 15, 60, 70): inf, (51, 15, 70, 70): inf}
rel_error_iron = 0.5
rel_error_bis = 0.5


fig = plt.figure()
i = 0
for key, value in conds.items():
    (_, degree, p, d) = key
    theta = np.matrix(np.arange(0., 180., degree))
    
    im_rec = reconstruct_image(im, theta, p, d)
    (boxes_iron, boxes_bis) = detect_bullets(im_rec, mu_iron, mu_bis, rel_error_iron, rel_error_bis)
    # print(i, key, len(boxes_bis), len(boxes_iron))

    if len(boxes_bis) == 10:
        i += 1
        print(i, key, value, len(boxes_bis), len(boxes_iron))

        plt.subplot(2, 3, i)
        plt.imshow(im_rec)
        plt.axis('off')
        plt.title(str(i))

    if i == 6:
        break

plt.show()
# %%

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

raw_data = pd.read_csv('EyeTrack-raw.tsv', sep='\t')
threshold = 500

raw_data = raw_data[raw_data["GazeEventDuration(mS)"] >= threshold]

# printing data
XCoord = raw_data["GazePointX(px)"].values 
YCoord = raw_data["GazePointY(px)"].values 
Recording = raw_data["RecordingTimestamp"].values
GazeDuration=raw_data["GazeEventDuration(mS)"].values

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

 
# create data

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
# nbins=300
# k = kde.gaussian_kde([XCoord,YCoord])
# xi, yi = np.mgrid[XCoord.min():XCoord.max():nbins*1j, YCoord.min():YCoord.max():nbins*1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
 
# # Make the plot
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
# plt.show()
 

# plt.hist2d(XCoord, YCoord, bins=(10, 10), cmap=plt.cm.jet)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# colors = np.random.uniform(15, 80, len(XCoord))

# ax.scatter(XCoord, YCoord, Recording)
# plt.show()

# printing data

## K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize



# Convert data to numpy array
X = np.column_stack((XCoord, YCoord, max(XCoord) *(Recording/max(Recording)), GazeDuration))
# print(max(XCoord) *(Recording/max(Recording)))

# # Initialize list to store SSE values for different k values
sse = []

# Fit k-means clustering algorithm for k values from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X[:,[0,1]])
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), sse)
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Specify the number of clusters
n_clusters = 4

# Apply k-means clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X[:,[0,1]])

# Get the cluster labels
labels = kmeans.labels_
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the clusters with different colors
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # you can define more colors if needed
for i in range(n_clusters):
    ax.scatter(X[labels==i,0], X[labels==i,1], X[labels==i,2], s=1000*(X[labels==i,3]/max(GazeDuration)), alpha=0.2, c=colors[i], label=f'Cluster {i+1}')
   

# Define a colormap
cmap = plt.cm.get_cmap('viridis')

# Normalize the data to the range [0, 1]
normalize = plt.Normalize(vmin=X[:,2].min(), vmax=X[:,2].max())

# Map the data values to colors in the colormap
colors = [cmap(normalize(value)) for value in X[:,2]]

print(len(colors))


for idx, x in enumerate(X):
    
    ax.plot(X[idx:idx+2,0], X[idx:idx+2,1], X[idx:idx+2,2], color= colors[idx],  alpha=0.5 )



# ax.plot(X[:,0], X[:,1], X[:,2], color= "r",  alpha=1 )
# Plot the centroids of each cluster
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='*', c='black', label='Centroids')

# plt.legend()

plt.show()


# How many regions?
# Using K-means we can cluster the data
# Elbow curve shows that 4 clusters is optimal.  
# This elbow curve uses x and y gazepoints to identiyfy where the distortion/inierta starts to decrease in a linear way

#how many are hevily used and when?
#All of the regions are used quite hevily but during different timeperiods, Bottom left seems to be start and finish while upper right 
# is used during the middle of the test. 








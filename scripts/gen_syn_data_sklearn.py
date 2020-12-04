import numpy as np
import pandas as pd
# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Needed for generating classification, regression and clustering datasets
import sklearn.datasets as dt

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

dataset_path_local = 'C:\\Users\\Stefan\\PycharmProjects\\thesis\\datasets\\iris_classification\\'

# Define the seed so that results can be reproduced
seed = 11
rand_state = 11

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "cyan", "magenta", "blue"])

plt_ind_list = np.arange(3) + 131

for i,x in zip([0.1, 1, 10, 1, 0], plt_ind_list):
    print(i,x)

# for class_sep, plt_ind in zip([0.1, 1, 10, 1, 0], plt_ind_list):
#     x, y = dt.make_classification(n_samples=1000,
#                                   n_features=4,
#                                   n_repeated=0,
#                                   class_sep=class_sep,
#                                   n_redundant=0,
#                                   random_state=rand_state)
#
#     plt.subplot(plt_ind)
#     my_scatter_plot = plt.scatter(x[:, 0],
#                                   x[:, 1],
#                                   c=y,
#                                   vmin=min(y),
#                                   vmax=max(y),
#                                   s=35,
#                                   cmap=color_map_discrete)
#     plt.title('class_sep: ' + str(class_sep))
#
# fig.subplots_adjust(hspace=0.3, wspace=.3)
# plt.suptitle('make_classification() With Different class_sep Values', fontsize=20)
# plt.show()


# x, y = dt.make_classification(  n_samples=1000,
#                                 n_features=4,
#                                 n_repeated=0,
#                                 class_sep=1,
#                                 n_redundant=0,
#                                 random_state=rand_state)
#
# plt.subplot(132)
# my_scatter_plot = plt.scatter(x[:, 0],
#                               x[:, 1],
#                               c=y,
#                               vmin=min(y),
#                               vmax=max(y),
#                               s=35,
#                               cmap=color_map_discrete)
# plt.title('class_sep: ' + str(1))
# plt.show()
#
# x, y = dt.make_classification( n_samples=1000,
#                                n_features=4,
#                                n_informative=4,
#                                n_redundant=0,
#                                n_repeated=0,
#                                n_classes=3,
#                                n_clusters_per_class=1,
#                                class_sep=2,
#                                scale=1.0,
#                                flip_y=0,
#                                random_state=11)
#
# plt.subplot(132)
# my_scatter_plot = plt.scatter(x[:, 0],
#                               x[:, 1],
#                               c=y,
#                               vmin=min(y),
#                               vmax=max(y),
#                               s=35,
#                               cmap=color_map_discrete)
# plt.title('class_sep: ' + str(1))
# plt.show()

centerbox = (0.1, 10.0)

x, label = dt.make_blobs(n_features=4,
                         n_samples=2000,
                         center_box=centerbox,
                         cluster_std=1.0,
                         random_state=rand_state)

plt.subplot(132)
my_scatter_plot = plt.scatter(x[:, 0],
                              x[:, 1],
                              c=label,
                              vmin=min(label),
                              vmax=max(label),
                              cmap=color_map_discrete)
plt.title('cluster_std: ' + str(1.0))
plt.show()
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import os

figs_dir = '../../Papers/Forecasting-With-Imperfect-No-Noise/NEW_Article/figures/'

N_1s = [int(1e7),int(5*1e6),int(1e6),int(1e5*5),int(1e5*2),int(1e5*1.5),int(1e5),int(1e4*7.5),int(1e4*5),int(1e4*2.5),int(1e4),int(1e3*5),int(1e3)][::-1]
N_2s = [int(1e7),int(5*1e6),int(2*1e6), int(1e6), int(1e5*5), int(1e5), int(1e4*7.5), int(1e4*5), int(1e4*2.5),int(1e4),int(1e3*5),int(1e3)][::-1]

if not os.path.isfile('./partSize.npy'):

    DATA = []

    for i in range(len(N_1s)):
        for j in range(len(N_2s)):
            N_1 = N_1s[i]
            N_2 = N_2s[j]
            K = pickle.load(open(f'./DataStructures/DS{N_1}/partFamily{N_2}.pkl','rb'))
            DATA += [[N_1, N_2, len(K.children)]]
            print(f'{N_1}, {N_2} measured')
    DATA = np.array(DATA)
    np.save('partSize.npy', DATA)
else:
    DATA = np.load('partSize.npy')



X, Y = np.meshgrid(N_1s,N_2s)

if not os.path.isfile('DTMat.npy'):
    DTMat = np.empty((len(N_2s),len(N_1s)))
    for i in range(len(N_2s)):
        for j in range(len(N_1s)):
            N_1 = N_1s[j]
            N_2 = N_2s[i]
            K = pickle.load(open(f'./DataStructures/DS{N_1}/partFamily{N_2}.pkl','rb'))
            DTMat[i][j] = len(K.children) 
            print(f'{N_1}, {N_2} measured')
    np.save('DTMat.npy', DTMat)
else:
    DTMat = np.load('DTMat.npy')

"""
Plotting
"""
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

# ax.plot_trisurf(X,Y,DTMat, color='#1f77b4')
ax.plot_surface(X.T,Y.T, DTMat.T, color='#1f77b4', alpha=0.3)

ax.scatter3D(DATA.T[0], DATA.T[1], DATA.T[2],s=35, color='#1f77b4', alpha=1)
ax.set_xlabel('Partition DATA',fontsize=14)
ax.set_ylabel(r'Sorting DATA',fontsize=14)
# ax.set_xticks(N_1s)
# ax.set_yticks(np.log(N_2s)/np.log(10))
# ax.set_zticks(np.linspace(12_000, 44_000, 10))
# ax.set_yticklabels([r'$10^5$',r'$2 \times 10 ^ 5$', r'$5 \times 10 ^5$', r'$10^6$',r'$5 \times 10^6$', r'$10^7$',r'$5 \times 10^7$', r'$10^8$' ])
ax.set_zlabel('Partition Count',fontsize=14)
ax.set_title('Partition Count for various models', fontsize=24)

plt.savefig(''.join([figs_dir,'/parsize.pdf']),bbox_inches='tight')

plt.show()

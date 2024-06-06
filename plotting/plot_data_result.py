import matplotlib.pyplot as plt
import numpy as np


figs_dir = '../../Papers/Forecasting-With-Imperfect-No-Noise/NEW_Article/figures/'

N_1s = [int(1e7),int(5*1e6),int(1e6),int(1e5*5),int(1e5*2),int(1e5*1.5),int(1e5),int(1e4*7.5),int(1e4*5),int(1e4*2.5),int(1e4),int(1e3*5),int(1e3)][::-1]
N_2s = [int(1e7),int(5*1e6),int(2*1e6), int(1e6), int(1e5*5), int(1e5), int(1e4*7.5), int(1e4*5), int(1e4*2.5),int(1e4),int(1e3*5),int(1e3)][::-1]
DATA = []

for i in range(len(N_1s)):
    for j in range(len(N_2s)):
        N_1 = N_1s[i]
        N_2 = N_2s[j]
        dataset =  np.load(f'./Results/{N_1}_Adjusted/{N_1}Adjusted{N_2}_50.npy')
        variance = np.var(dataset)
        DATA += [[N_1,N_2, np.mean(dataset), variance]]
DATA = np.array(DATA)

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

X, Y = np.meshgrid(N_1s,N_2s)

DTMat = np.empty((len(N_2s),len(N_1s)))
for i in range(len(N_2s)):
    for j in range(len(N_1s)):
        N_1 = N_1s[j]
        N_2 = N_2s[i]
        DTMat[i][j] = np.mean(np.load(f'./Results/{N_1}_Adjusted/{N_1}Adjusted{N_2}_50.npy'))

ax.plot_surface(Y.T,X.T, DTMat.T, color='#1f77b4', alpha=0.3)

ax.scatter3D(DATA.T[1], DATA.T[0], DATA.T[2],s=30,alpha=1)
ax.set_ylabel('Partition DATA',fontsize=14)
ax.set_xlabel(r'Sorting DATA',fontsize=14)
#ax.set_xticks(N_1s)
#ax.set_yticks(N_2s)
ax.set_zlabel('Forecasting Separation Times',fontsize=14)
ax.set_title('Separation Times for various sized models', fontsize=24)

plt.savefig(''.join([figs_dir,'/sepTime.pdf']))
fig2, ax2 = plt.subplots()

def split_By_N_1(arr):
    N1s = np.unique(arr.T[0])
    N2s = np.unique(arr.T[1])
    num_uniquesN1 = len(N1s)

    New_Matrix = [ [] for _  in range(num_uniquesN1) ]

    for j in range(num_uniquesN1):
        for row in arr:
            if row[0] == N1s[j]:
                New_Matrix[j] += [[row[1],row[2]]]
    New_Matrix = np.array([np.array(arr) for arr in New_Matrix])
    return New_Matrix
A = split_By_N_1(DATA)
for i in range(len(A)):
    x = A[i].T[0]
    y = A[i].T[1]
    ax2.plot(x,y, label= str(N_1s[i]))


ax2.legend(title="Partition Size")


plt.savefig(''.join([figs_dir,'/sepTime2D.pdf']))


plt.show()

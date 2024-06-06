import os 
import numpy as np
import matplotlib.pyplot as plt

N_1 = 100_000
N_2 = int(1e8)
files = ['Adjusted', 'DynamicPF', 'StaticPF', 'Unadjusted']

adj = [ 
       np.mean(np.load(f'./Results/PFComparison/{N_1}_Adjusted/{N_1}Adjusted{N_2}_{ne}.npy')) 
       for ne in [1,5,10,15,20,50,100,150]
                ]

unj = np.ones(8) * np.mean(np.load(f'./Results/PFComparison/{N_1}_Unadjusted/{N_1}Unadjusted{N_2}.npy'))

static = [ 
       np.mean(np.load(f'./Results/PFComparison/{N_1}_StaticPF/{N_1}PF_STATIC{N_2}_{ne}.npy')) 
       for ne in [1,5,10,15,20,50,100,150]
                ]

dyn = [ 
       np.mean(np.load(f'./Results/PFComparison/{N_1}_DynamicPF/{N_1}PF_Dynamic{N_2}_{ne}.npy')) 
       for ne in [1,5,10,15,20,50,100,150]
                ]
N_E = [1,5,10,15,20,50,100,150]
fig, ax = plt.subplots()


ax.set_title('Results', fontsize=20)
ax.set_xlabel('Number of Ensembles',fontsize=12)
ax.set_ylabel('Mean Separation Time', fontsize=12)
ax.set_xticks(np.arange(0,160,10))
ax.set_yticks(np.arange(0,4,0.1))

ax.plot(N_E, adj, label='Adjusted', color='blue')
ax.plot(N_E,unj, label="Unadjusted",color='green')
ax.plot(N_E,static, label = 'Static PF', color='red')
ax.plot(N_E,dyn, label='Dynamic PF', color='purple')

ax.scatter(N_E, adj, color='blue')
ax.scatter(N_E, static, color='red')
ax.scatter(N_E, dyn, color='purple')


ax.legend()

plt.savefig('Results.pdf',bbox_inches='tight')

plt.show()





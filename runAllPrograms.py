PARTITION_DATA = [int(1e7),int(5*1e6),int(1e6),int(1e5*5),int(1e5*2),int(1e5*1.5),int(1e5),int(1e4*7.5),int(1e4*5),int(1e4*2.5),int(1e4),int(1e3*5),int(1e3)]
SORTING_DATA = [int(1e7),int(5*1e6),int(2*1e6), int(1e6), int(1e5*5), int(1e5), int(1e4*7.5), int(1e4*5), int(1e4*2.5),int(1e4),int(1e3*5),int(1e3)]

N_E = 50

# Build data
from GenerateDataStructures import buildAllModels
buildAllModels(PARTITION_DATA, SORTING_DATA)

# Get Results
from genResults_varData import GenerateResults_Var_DATA
GenerateResults_Var_DATA(PARTITION_DATA, SORTING_DATA, N_E)



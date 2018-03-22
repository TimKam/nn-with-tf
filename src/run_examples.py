from network_0 import run_network_0
from network_1 import run_network_1
from network_2 import run_network_2

data_dir = '../Datasets/cifar-10-batches-py/'

# 1-layer network, different learning rates:
# run_network_0(data_dir, '../graphs/network0/results/1000_200_0.01', 200, 1000, 0.01)
# run_network_0(data_dir, '../graphs/network0/results/1000_200_0.0001', 200, 1000, 0.0001)
# run_network_0(data_dir, '../graphs/network0/results/1000_200_1', 200, 1000, 1)

# 1-layer network, different batch sizes:

# run_network_0(data_dir, '../graphs/network0/results/1000_10_0.01', 10, 1000, 0.01)
# run_network_0(data_dir, '../graphs/network0/results/1000_2000_0.01', 2000, 1000, 0.01)

# 1-layer network, "good" parameters:
# run_network_0(data_dir, '../graphs/network0/results/10000_300_0.01', 300, 10000, 0.01)

# 2-layered network:
# run_network_1(data_dir, '../graphs/network1/results/10000_300_0.01', 300, 10000, 0.01)
# run_network_1(data_dir, '../graphs/network1/results/50000_300_0.01', 300, 50000, 0.01)

# convolutional network, will take a long time to run (tens of hours on conventional laptop)
# run_network_2(data_dir, '../graphs/network2/results/50000_300_0.01', 300, 50000, 0.01)

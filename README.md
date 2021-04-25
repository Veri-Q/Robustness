VeriQ
===
Phython Toolbox for Robustness Verification of Quantum Classifiers 

This toolbox is implemented on Google’s TensorFlow Quantum and can verify the robustness of quantum machine learning algorithms with respect to a small disturbance of noises, derived from the surrounding environment.

### Requirements 
---
This toolbox makes use of [Numpy](https://numpy.org) and an SDP solver — [CVXPY](https://www.cvxpy.org/): Python Software for Disciplined Convex Programming. 

### Installation for Unix, Linux (Ubuntu 18.04 as the example)
---
1) The installation of VeiQ requires BLAS and LAPACK. Cmake and pip3 are also needed.
```sh
sudo apt install -y libblas-dev liblapack-dev cmake python3-pip
```
2) Because the default version of Python in Ubuntu 18.04 is Python3.6, we should install Numpy first.
```sh
pip3 install numpy
```
3) Install CVXPY.
```sh
pip3 install cvxpy
```
4) In addition, our demostration file `batch_check.py` uses a Python library PrettyTable for printing a format table.
```sh
pip3 install prettytable
```
5) Clone or download the VeriQ toolbox from (https://github.com/j68249959/VeriQ). All files must be saved in the same location. 

### Running Tests and Examples
---
###### Try the follwing commands in bash for robustness verification of four quantum classifiers.

1) Quantum Bits Classifications
```sh
python3 batch_check.py binary_cav.mat 1e-3 4 mixed
```
2) Quantum Phase Recognition 
```sh
python3 batch_check.py phase_recong_cav.mat 1e-4 4 mixed
```
3) Cluster Excitation Detection 
```sh
python3 batch_check.py excitation_cav.mat 1e-4 4 mixed
```
4) The Classification of MNIST
```sh
python3 batch_check.py mnist_cav.mat 1e-4 4 pure
```
### Experimental Results
---
######After running tests, you will get the following results. It is worth noting that the verification time is depending on the performance of your computer devices.
1) Quantum Bits Classifications
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/Binary.png)
2) Quantum Phase Recognition 
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/Phase.png)
3) Cluster Excitation Detection 
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/Excitation.png)
4) The Classification of MNIST
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/MNIST.png)

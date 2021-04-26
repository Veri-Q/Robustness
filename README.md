VeriQ
===
Phython Toolbox for Robustness Verification of Quantum Classifiers 

This toolbox is implemented on Python and can verify the robustness of quantum machine learning algorithms with respect to a small disturbance of noises, derived from the surrounding environment.

### Requirements 
---
This toolbox makes use of [Numpy](https://numpy.org) and an SDP solver — [CVXPY](https://www.cvxpy.org/): Python Software for Disciplined Convex Programming. 

### Installation
---
###### VeriQ can be installed on Unix and Linux. The following installation instruction is based on Ubuntu 18.04.

1) The installation of VeriQ requires BLAS and LAPACK. Cmake and pip3 are also needed.
```sh
sudo apt install -y libblas-dev liblapack-dev cmake python3-pip
```
2) Because the default version of Python in Ubuntu 18.04 is Python3.6, Numpy should be installed first.
```sh
pip3 install numpy
```
3) Install CVXPY.
```sh
pip3 install cvxpy
```
4) In addition, our demostration file `batch_check.py` uses a Python library PrettyTable for printing a format table summerizing the numerical results.
```sh
pip3 install prettytable
```
5) Clone or download the VeriQ toolbox from (https://github.com/j68249959/VeriQ). All files must be saved in the same location.
```sh
git clone https://github.com/j68249959/VeriQ
```

### Running Tests and Examples
---
To implement robustness verification in VeriQ, we assume that the user has already well trained a quantum classifier which consists of a quantum circuit with a measurement at the end. The quantum circuit and the trainning data have been saved as a numpy data file.

Then try the follwing commands in bash for robustness verification of four quantum classifiers.
1) Quantum Bits Classifications
```sh
python3 batch_check.py binary_cav.npz 1e-3 4 mixed
```
The last four arguments are inputted by users. The first one `binary_cav.npz` is a numpy data file that consists of a (well-trained) quantum classifier and coresponding training dataset. The numpy data file can be obatined by the data of the classifiers trained on the platform --- [Tensorflow Quantum](https://www.tensorflow.org/quantum/) of Google. The second argument `1e-3` is the unit of the robustness parameter. The third argument `4` represents the number of robustness verification with increasing robustness parameters by the unit `1e-3`, i.e., `1e-3`, `2e-3`,`3e-3`, `4e-3`-robustness to be checked one by one. The last one indicates the considering quantum data in robustness verification, where  `mixed`  and `pure` refer to mixed states and pure states, respectively.

2) Quantum Phase Recognition 
```sh
python3 batch_check.py phase_recog_cav.npz 1e-4 4 mixed
```
3) Cluster Excitation Detection 
```sh
python3 batch_check.py excitation_cav.npz 1e-4 4 mixed
```
4) The Classification of MNIST
```sh
python3 batch_check.py mnist_cav.npz 1e-4 4 pure
```
### Experimental Results
---
###### After running tests, you will get the following results. It is worth noting that the verification time is depending on the performance of your computer devices.
1) Quantum Bits Classifications
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/Binary.png)
2) Quantum Phase Recognition 
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/Phase.png)
3) Cluster Excitation Detection 
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/Excitation.png)
4) The Classification of MNIST
![avatar](https://github.com/j68249959/VeriQ/blob/main/Experimental%20Results/MNIST.png)

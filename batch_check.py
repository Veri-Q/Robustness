from Qrobustness import RobustnessVerifier, PureRobustnessVerifier
from scipy.io import loadmat
from prettytable import PrettyTable
from sys import argv

data_file = str(argv[1])
eps = float(argv[2])
n = int(argv[3])
state_flag = str(argv[4])

if state_flag == 'mixed':
    verifier = RobustnessVerifier
else:
    verifier = PureRobustnessVerifier

DATA = loadmat(data_file)

ac = PrettyTable()
time = PrettyTable()
ac.add_column('epsilon', ['Robust Bound', 'Robustness Algorithm'])
time.add_column('epsilon', ['Robust Bound', 'Robustness Algorithm'])
for j in range(n):
    c_eps = eps * (j + 1)
    ac_temp, time_temp = verifier(
        DATA['kraus'].astype(complex),
        DATA['O'].astype(complex),
        DATA['data'].astype(complex),
        DATA['label'].T[:,0],
        c_eps)
    
    ac.add_column('{:e}'.format(c_eps), [
        '{:.2f}'.format(ac_temp[0] * 100), 
        '{:.2f}'.format(ac_temp[1] * 100)])
    time.add_column('{:e}'.format(c_eps), [
        '{:.4f}'.format(time_temp[0]),
        '{:.4f}'.format(time_temp[1])])
 
print('Robust Accuracy (in Percent)')
print(ac)
print('Verification Times (in Seconds)')
print(time)

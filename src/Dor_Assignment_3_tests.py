import subprocess
import numpy as np


def main():
    input_str = ""
    num_equations = 3
    num_iter = 10
    for i in range(num_iter):
        input_str += str(num_equations) + "\r\n"
        mat = np.random.rand(num_equations, num_equations + 1).astype('f', order='C') * 10
        num_spaces = np.random.randint(1, 5, (num_equations, 1))
        var_ind_to_delete = np.random.randint(0, 2, (num_equations, 1))
        for j in range(mat.shape[0]):
            for k in range(mat.shape[1] - 1):
                if k == var_ind_to_delete[j].item():
                    continue
                if k == 0:
                    input_str += str(mat[j, k]) + ' ' * int(num_spaces[j].item()) + ' * x + '
                elif k == 1:
                    input_str += str(mat[j, k]) + ' ' * int(num_spaces[j].item()) + ' * y + '
                elif k == 2:
                    input_str += str(mat[j, k]) + ' ' * int(num_spaces[j].item()) + ' * z'
            input_str += ' ' * int(num_spaces[j].item()) + '=' + str(mat[j, -1])
            input_str += "\r\n"
        if i < (num_iter - 1):
            input_str += "y\r\n"
        else:
            input_str += "n\r\n"
    p = subprocess.Popen(
        [r"C:\Users\galun\Desktop\Ran\Dor_Assignments_C\Assignment_3_Dor\X64\Debug\Dor_Assignment_3.exe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        outs, errs = p.communicate(input=input_str.encode("ascii"))
        p.terminate()
    except subprocess.TimeoutExpired:
        p.kill()
        outs, errs = p.communicate()
    print(p.returncode)
    print(outs.decode())


if __name__ == "__main__":
    main()

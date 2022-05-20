import subprocess
import numpy as np


def main():
    input_str = ""
    num_equations = 3
    for i in range(1):
        input_str += str(num_equations) + "\r\n"
        mat = np.random.rand(num_equations, num_equations).astype('f', order='C') * 100
        # num_spaces = np.zeros((num_equations, 1))
        num_spaces = np.random.randint(1, 2, (num_equations, 1))
        for j in range(mat.shape[0]):
            for k in range(mat.shape[1]):
                if k == 0:
                    input_str += str(mat[j, k]) + ' ' * int(num_spaces[j].item()) + ' * x + '
                elif k == 1:
                    input_str += str(mat[j, k]) + ' ' * int(num_spaces[j].item()) + ' * y + '
                elif k == 2:
                    input_str += str(mat[j, k]) + ' ' * int(num_spaces[j].item()) + ' * z'
            input_str += "\r\n"
    print(input_str)
    p = subprocess.Popen(
        [r"C:\Users\galun\Desktop\Ran\Dor_Assignments_C\Assignment_3_Dor\X64\Debug\Dor_Assignment_3.exe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        outs, errs = p.communicate(input=input_str.encode("ascii"))
    except subprocess.TimeoutExpired:
        p.kill()
        outs, errs = p.communicate()
    print(outs.decode())


if __name__ == "__main__":
    main()

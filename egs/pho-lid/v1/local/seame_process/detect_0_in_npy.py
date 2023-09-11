import os
import numpy as np

def detect_0_in_tensor(ten):
    check_all_0 = np.all(ten==0.0000e+00)
    return check_all_0

if __name__ == "__main__":
    # npy1 = np.array([[0, 0, 0.0000e+00], [0, 0, 0.0000e+00]])
    # npy2 = np.array([[0, 0, 0.0000e+00], [1, 0, 0.0000e+00]])

    # print(f'npy1: {detect_0_in_tensor(npy1)}, npy2: {detect_0_in_tensor(npy2)}')


    data_root = "./local/seame_split/"
    data_lists = ["/pure/feat2lang_train_final.txt", 
        "/pure/feat2lang_test_final.txt", 
        "/cat/feat2lang_train_final.txt", 
        "/cat/feat2lang_test_final.txt"]

    data_lists_new = ["/pure/feat2lang_train_new.txt", 
        "/pure/feat2lang_test_new.txt", 
        "/cat/feat2lang_train_new.txt", 
        "/cat/feat2lang_test_new.txt"]

    data_lists_0 = ["/pure/feat2lang_train_0.txt", 
        "/pure/feat2lang_test_0.txt", 
        "/cat/feat2lang_train_0.txt", 
        "/cat/feat2lang_test_0.txt"]

    for data_list, new_data_list, zero_data_list in zip(data_lists, data_lists_new, data_lists_0):
        with open(data_root + data_list, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            path = line.split(sep='\t')[0]
            
            npy = np.load(path).ravel()

            if detect_0_in_tensor(npy):
                with open(data_root + zero_data_list, 'a') as f0:
                    f0.write(line)
                print(f'result:{detect_0_in_tensor(npy)}, npy: {npy}')
                
            else:
                with open(data_root + new_data_list, 'a') as f1:
                    f1.write(line)
                
        with open(data_root + new_data_list, 'r') as f1:
            new_lines = len(f1.readlines())

        if len(lines) == new_lines:
            print(f'No chages to file {data_list}.')

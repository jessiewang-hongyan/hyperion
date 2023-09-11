import os

if __name__=='__main__':
    file_path = './data/seame/'
    print(len(os.listdir(file_path)))

    
    # for fname in os.listdir(file_path):
    #     with open(file_path + fname + '/cat/new_data_label_list.txt', 'r') as f:
    #         lines = f.readlines()
    #     cat_names = [x.split(sep='\t')[0] for x in lines]

    #     # for cat_name in cat_names:
    #     #     os.remove(file_path + fname + '/cat/' + cat_name)
    #     # os.remove(file_path + fname + '/cat/new_data_label_list.txt')
    
    #     with open(file_path + fname + '/processed/new_segment2lang.txt', 'r') as f:
    #         lines = f.readlines()
    #     cat_names = [x.split(sep='\t')[0] for x in lines]

    #     for cat_name in cat_names:
    #         os.remove(file_path + fname + '/processed/' + cat_name)
    #     # os.remove(file_path + fname + '/processed/new_segment2lang.txt')
    #     # os.remove(file_path + fname + '/processed/new_wav2lang.txt')

    #     npy_names = [cn.replace('.wav', '.npy') for cn in cat_names]

    #     with open(file_path + fname + '/processed/feat2lang.txt', 'r') as f:
    #         lines = f.readlines()
    #     feat2lang_npy_names = [x.split(sep='\t')[0] for x in lines]

    #     new_lines =[]
    #     for fn, line in zip(feat2lang_npy_names, lines):
    #         if fn not in npy_names:
    #             new_lines.append(lines)

    #     with open(file_path + fname + '/processed/feat2lang_new.txt', 'w') as f:
    #         for line in new_lines:
    #             f.write(line)

    #     print(f'{fname} done!')
    # print('Cleaning Done.')
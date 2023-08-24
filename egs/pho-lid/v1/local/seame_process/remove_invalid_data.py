import os

if __name__=='__main__':
    file_path = './data/seame/merge/'
    new_file_path = './local/seame_split/'
    # os.mkdir(new_file_path)
    # os.mkdir(new_file_path+'/cat')
    # os.mkdir(new_file_path+'/pure')

    filenames = ['pure/feat2lang_train.txt', 'pure/feat2lang_test.txt', 'cat/feat2lang_train.txt', 'cat/feat2lang_test.txt']
    for fname in filenames:
        with open(file_path+fname, 'r') as f:
            lines = f.readlines()
        npy_names = [x.split(sep='\t')[0] for x in lines]

    
        with open(new_file_path + fname, 'a') as f1:
            for npy, line in zip(npy_names, lines):
                if os.path.exists(npy):
                    f1.write(line)
        print(f'{fname} done!')
    print('Cleaning Done.')
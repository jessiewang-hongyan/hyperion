import os

if __name__ == '__main__':
    # read in spk split
    train_spk_split = './local/seame_split/spk_train.txt'
    test_spk_split = './local/seame_split/spk_test.txt'

    with open(train_spk_split, 'r') as f:
        train_spks = f.readlines()
    train_spks = [x.strip() for x in train_spks]

    with open(test_spk_split, 'r') as f:
        test_spks = f.readlines()
    test_spks = [x.strip() for x in test_spks]

    merged_dir = './local/seame_split/'
    for folder_path in [merged_dir, merged_dir + '/cat/', merged_dir + '/pure/']:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    merged_cat_txt_train = merged_dir + '/cat/feat2lang_train.txt'
    merged_cat_txt_test = merged_dir + '/cat/feat2lang_test.txt'
    merged_pure_txt_train = merged_dir + '/pure/feat2lang_train.txt'
    merged_pure_txt_test = merged_dir + '/pure/feat2lang_test.txt'

    # clear old version
    for file_name in [merged_cat_txt_train, merged_cat_txt_test, merged_pure_txt_train, merged_pure_txt_test]:
        if os.path.exists(file_name):
            os.remove(file_name)

    for dir_name in os.listdir("data/seame/"):
        if not dir_name == 'merge':
            file_path = "/export/fs05/ywang793/hyperion/egs/pho-lid/v1/data/seame/" + dir_name
            
            if dir_name in train_spks:
                merged_pure_txt = merged_pure_txt_train
                merged_cat_txt = merged_cat_txt_train
            elif dir_name in test_spks: 
                merged_pure_txt = merged_pure_txt_test
                merged_cat_txt = merged_cat_txt_test
            
            if os.path.exists(file_path+'/pure_processed/feat2lang.txt'):
                input_file = file_path + '/pure_processed/feat2lang.txt'
                with open(input_file, 'r') as file:
                    lines = file.readlines()

                with open(merged_pure_txt, 'a') as file1:
                    file1.writelines(lines)

            if os.path.exists(file_path+'/processed/feat2lang.txt'):
                input_file = file_path + '/processed/feat2lang.txt'
                with open(input_file, 'r') as file:
                    lines = file.readlines()

                with open(merged_cat_txt, 'a') as file1:
                    file1.writelines(lines)

    print('Merge SEAME Done.')

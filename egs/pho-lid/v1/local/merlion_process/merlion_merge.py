import os

if __name__ == '__main__':
    # read in spk split
    train_spk_split = './local/merlion_split/spk_train.txt'
    test_spk_split = './local/merlion_split/spk_test.txt'
    feat2lang_file = '../../../../merlion/train/cat_processed/feat2lang.txt'

    with open(train_spk_split, 'r') as f:
        train_spks = f.readlines()
    train_spks = [x.strip() for x in train_spks]
    print(train_spks)

    with open(test_spk_split, 'r') as f:
        test_spks = f.readlines()
    test_spks = [x.strip() for x in test_spks]
    print(test_spks)

    with open(feat2lang_file, 'r') as f:
        feat2langs = f.readlines()
    print(len(feat2langs))

    merged_dir = './local/merlion_split/'

    merged_pure_txt_train = merged_dir + '/feat2lang_cat_train_new.txt'
    merged_pure_txt_test = merged_dir + '/feat2lang_cat_test_new.txt'

    # clear old version
    for file_name in [merged_pure_txt_train, merged_pure_txt_test]:
        if os.path.exists(file_name):
            os.remove(file_name)

    for feat_line in feat2langs:
        if len(feat_line.split()) > 1:
            feat_name = feat_line.split()[0]
            spk_name = feat_name.split(sep='_')[2]
            # print(f'feat_name: {feat_name}, spk_name: {spk_name}')
            if spk_name in train_spks:
                merged_pure_txt = merged_pure_txt_train
            elif spk_name in test_spks:
            # else: 
                merged_pure_txt = merged_pure_txt_test
            else:
                print(spk_name)

            with open(merged_pure_txt, 'a') as file1:
                file1.write(feat_line)

    print('Merge MERLIon Done.')

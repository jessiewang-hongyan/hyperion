import os
import json
import random
import math

if __name__ == '__main__':
    data_list = 'data/merlion/train/processed/feat2lang.txt'
    spk_file = 'local/merlion_split/spk_file.json'
    spk_train_split_file = 'local/merlion_split/spk_train.txt'
    spk_test_split_file = 'local/merlion_split/spk_test.txt'

    train_ratio = 0.5

    # Data to be written
    spk_dict = dict()

    with open(data_list, 'r') as f:
        lines = f.readlines()

    for x in lines:
        if len(x.split()) > 1:
            spk_rec = x.split()[0]
            if spk_rec[-4:] == '.npy':
                spk = spk_rec.split(sep='_')[1]
                if spk in spk_dict.keys():
                    spk_dict[spk].append(spk_rec)
                else:
                    spk_dict[spk] = [spk_rec]

    
    # Serializing json
    json_object = json.dumps(spk_dict, indent=4)
    
    # clear old version
    if os.path.exists(spk_file):
        os.remove(spk_file)

    # Writing to sample.json
    with open(spk_file, "w") as outfile:
        outfile.write(json_object)

    # Split records according to spk
    spk_list = list(spk_dict.keys())
    random.shuffle(spk_list)

    split_idx = math.floor(len(spk_list) * train_ratio)
    train_spk_list = spk_list[:split_idx]
    test_spk_list = spk_list[split_idx:]

    # clear old version
    for file_name in [spk_train_split_file, spk_test_split_file]:
        if os.path.exists(file_name):
            os.remove(file_name)

    # Write into spk files
    with open(spk_train_split_file, 'a') as f:
        for name in train_spk_list:
            f.write(name+'\n')

    with open(spk_test_split_file, 'a') as f:
        for name in test_spk_list:
            f.write(name+'\n')
                
    # Print result to log
    print('MERLIon train spk split done.')
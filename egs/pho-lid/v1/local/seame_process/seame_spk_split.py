import os
import json
import random
import math

if __name__ == '__main__':
    data_folder = 'data/seame/'
    spk_file = 'local/seame_split/spk_file.json'
    train_split_file = 'local/seame_split/spk_train.txt'
    test_split_file = 'local/seame_split/spk_test.txt'
    train_ratio = 0.5

    # Data to be written
    spk_dict = dict()

    for spk_rec in os.listdir(data_folder):
        spk = spk_rec.split(sep='_')[0][-5:-1]
        if spk in spk_dict.keys():
            spk_dict[spk].append(spk_rec)
        else:
            spk_dict[spk] = [spk_rec]

    # print(spk_dict)
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
    for file_name in [train_split_file, test_split_file]:
        if os.path.exists(file_name):
            os.remove(file_name)

    # Write into files
    with open(train_split_file, 'a') as f:
        for name in train_spk_list:
            for rec in spk_dict[name]:
                f.write(rec+'\n')

    with open(test_split_file, 'a') as f:
        for name in test_spk_list:
            for rec in spk_dict[name]:
                f.write(rec+'\n')
                
    # Print result to log
    print('SEAME spk split done.')
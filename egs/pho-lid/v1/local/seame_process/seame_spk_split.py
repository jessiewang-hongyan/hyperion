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

    # pure_empty_list = ['UI26MAZ_0104', 'NI02FAX_0101', 'UI14MAZ_0104', 'UI15FAZ_0104', 'UI07FAZ_0102', 'NI50FBQ_0101', 'UI10FAZ_0103', 'UI14MAZ_0105', 'UI04FAZ_0105', 'UI05MAZ_0105', 'UI21MAZ_0104', 'UI28FAZ_0104', 'NI64FBQ_0101', '20NC40FBQ_0101', 'NI61FBP_0101', '05NC09FAX_0101', 'UI12FAZ_0103', '03NC05FAX_0101', 'NI09FBP_0101', 'UI10FAZ_0102', 'NI05MBQ_0101', 'UI19MAZ_0102', '33NC37MBP_0101', '04NC07FBX_0101', '45NC22MBQ_0101', 'NI23FBQ_0101', '21NC42MBQ_0101', '05NC10MAY_0201', 'NI26FBP_0101', 'NI22FBP_0101', 'UI01FAZ_0105', 'UI02FAZ_0104', '30NC48FBP_0101', 'NI43FBP_0101', 'UI09MAZ_0101', '27NC47MBQ_0101', 'UI04FAZ_0101', 'UI05MAZ_0101', 'NI42FBQ_0101', '30NC49FBQ_0101', 'NI46FBQ_0101', 'UI25FAZ_0104', 'UI27FAZ_0101', 'UI17FAZ_0103', 'UI23FAZ_0101', 'UI13FAZ_0103', 'UI20MAZ_0104', 'UI22MAZ_0101', 'UI29FAZ_0104', 'UI16MAZ_0103', 'NI65MBP_0101', 'UI26MAZ_0101', 'UI24MAZ_0104', '14NC28MBQ_0101', 'UI20MAZ_0103', 'NI57FBQ_0101', 'UI29FAZ_0103', 'UI16MAZ_0104', 'UI14MAZ_0101', 'UI24MAZ_0103', 'UI25FAZ_0103', 'UI15FAZ_0101', 'UI17FAZ_0104', 'NI52MBQ_0101', 'NI08FBP_0201', 'UI18MAZ_0101', 'UI13FAZ_0104', 'UI11FAZ_0101', 'NI01MAX_0101', 'UI02FAZ_0103', '08NC15MBP_0101', '17NC33FBP_0101', '33NC43FBQ_0101', '02NC03FBX_0201', 'UI01FAZ_0108', 'NI10FBP_0101', 'UI08MAZ_0102', '20NC39MBP_0101', '22NC44MBQ_0101', 'NI15FBQ_0101', 'UI10FAZ_0105', '19NC38FBQ_0101', 'UI19MAZ_0105', '16NC31FBP_0101', '12NC24FBQ_0101', 'NI32FBQ_0101', '05NC09FAX_0201']
    # processed_empty = ['UI26MAZ_0104', 'NI02FAX_0101', 'UI14MAZ_0104', 'UI15FAZ_0104', 'UI07FAZ_0102', 'NI50FBQ_0101', 'UI10FAZ_0103', 'UI14MAZ_0105', 'UI04FAZ_0105', 'UI05MAZ_0105', 'UI04FAZ_0101', 'NI52MBQ_0101', 'NI01MAX_0101']

    # Write into files
    with open(train_split_file, 'a') as f:
        for name in train_spk_list:
            for rec in spk_dict[name]:
                # if rec not in pure_empty_list:
                    f.write(rec+'\n')

    with open(test_split_file, 'a') as f:
        for name in test_spk_list:
            for rec in spk_dict[name]:
                # if rec not in pure_empty_list:
                    f.write(rec+'\n')
                
    # Print result to log
    print('SEAME spk split done.')
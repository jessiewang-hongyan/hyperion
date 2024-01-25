import os

if __name__ == '__main__':
    merged_dir = './local/seame_split/'
    merged_cat_txt_test = merged_dir + '/cat/feat2lang_test_final.txt'
    rttm_file = "./inference/outputs/seame_ground_truth_channel0.rttm"
    lang_map = ["English", "Mandarin"]

    lines = []
    with open(merged_cat_txt_test, 'r') as file:
            rows = file.readlines()

            # read each line of record
            for row in rows:
                audio, labels, _ = row.strip().split("\t")
                # audio = audio.split("/")[-1].replace(".npy", "")
                audio = audio.replace(".npy", "")
                labels = labels.replace("[", "").replace("]", "").split(", ")
                labels = [int(float(x)) for x in labels]

                start = 0
                interval = 0.4
                for label in labels:
                    fields = ['SPEAKER', audio, '0', str(start), str(interval), '<NA>', '<NA>', str(lang_map[label]), '<NA>', '<NA>']
                    line = ' '.join(fields)
                    lines.append(line)
                    start += interval

    # write into rttm file
    with open(rttm_file, 'wb') as file:
        for line in lines:
            file.write(line.encode('utf-8'))
            file.write(b'\n')



    print('SEAME rttm conversion done.')

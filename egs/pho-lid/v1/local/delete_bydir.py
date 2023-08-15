import os
import glob


if __name__ == '__main__':
    dataset = 'data/seame'
    for dir in os.listdir(dataset):
        print(dir)

        person_name = "data/seame/" + dir

        # files = glob.glob(person_name+'/seg/*')
        # for f in files:
        #     os.remove(f)

        # files = glob.glob(person_name+'/cat/*')
        # for f in files:
        #     os.remove(f)


        files = glob.glob(person_name+'/pure/*')
        for f in files:
            os.remove(f)
            

        # file = glob.glob(person_name+'/data_label_list.txt')
        # os.remove(f)
        # files = glob.glob(person_name+'/pure_processed/*')
        # for f in files:
        #     os.remove(f)

        # os.rmdir(person_name+'/pure_processed')
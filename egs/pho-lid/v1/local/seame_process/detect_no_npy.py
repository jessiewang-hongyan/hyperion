import os

if __name__ == '__main__':
    data_root = './data/seame/'
    pure_empty = []
    processed_empty = []

    for dir_name in os.listdir(data_root):
        if not dir_name == 'merge':
            is_pure_empty = not any([x.endswith('.npy') for x in os.listdir(data_root+dir_name+'/pure/')])
            is_processed_empty = not any([x.endswith('.npy') for x in os.listdir(data_root+dir_name+'/processed/')])

            if is_pure_empty:
                pure_empty.append(dir_name)
        
            if is_processed_empty:
                processed_empty.append(dir_name)

    print(f'pure_empty: {pure_empty}')
    print(f'processed_empty: {processed_empty}')
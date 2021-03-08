import pickle
import os
from tqdm import tqdm




def dict_merge(first_dic,second_dic):
    intersect_keys = set(first_dic.keys()).intersection(set(second_dic.keys()))
    not_in_x = second_dic.keys() - intersect_keys
    for key in intersect_keys:
        first_dic[key].update(second_dic[key])
    for key in not_in_x:
        first_dic[key] = second_dic[key] 




if __name__ == '__main__':
    path_first = 'C:\\Users\\Omer\\NLP\\NLP_Project\\dpr\\paragraph_matcher\\index\\inv_index_5'
    first_index = pickle.load(open(path_first,'rb'))
    path_index = 'C:\\Users\\Omer\\NLP\\NLP_Project\\dpr\\paragraph_matcher\\index'
    for file in os.listdir(path_index):
        if(file[:3]=='inv' and len(file)>11):
            file_to_merge_path = path_index+f'\\{file}'
            print(f'loading {file}')
            second_file = pickle.load(open(file_to_merge_path,'rb'))
            print(f'starting to merge {file}')
            dict_merge(first_index,second_file)
            del second_file
    print('starting to write final index')
    with open('final_dict','wb') as out:
                pickle.dump(first_index,out)
    
import json
import os 
import argparse
import glob
import shutil 
import random
import re

global parser

parser = argparse.ArgumentParser(description="Detectron 2 script formater. Takes 1 viaproject json file as input")
parser.add_argument(
        "--via_file",
        metavar="FILE",
        help="path to via file",
    )
parser.add_argument(
        '--images_src',
        help='Cropped images path'
)

parser.add_argument(
        '--images_dst',
        help='Dest of coppied images (test and train folders will be created)'
)
args = parser.parse_args()


def get_individual_files(via_file):
    
    print('On get_individual_files')
    with open(via_file) as json_file:
        data = json.load(json_file)
        # for reading nested data [0] represents
        # the index value of the list
        dict_instances = {}
        for key, value in data.items():
        #dict_new = {}
        #dict_new['version'] = '4.2.9'
        #dict_new['flags'] = ''
            #print(f'{key}: {value}')
            if value["regions"]:
                dict_instances[key] = value
                #print(f'regions: {value["regions"]}')
    return dict_instances

def generate_files(dict_instances):
    print('On generate_files')
    #print('New Dict: ', dict_instances)\
    for key, value in dict_instances.items():
        #print(f'\n {key} \n')
        dict_instance = {}
        dict_instance['version'] = '4.2.9'
        dict_instance['flags'] = {}
        dict_instance['shapes'] = []
        #print(f'{key}: {value["regions"]}')
        for idx, region in enumerate(value['regions']):
            dict_instance_sub = {}
            dict_instance_sub['label'] = 'disease'
            #print(idx, ':', region['shape_attributes']['all_points_x'])
            points = list(map(list,list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))))
            dict_instance_sub['points'] = points
            dict_instance_sub['group_id'] = None
            dict_instance_sub['shape_type'] = 'polygon'
            dict_instance['shapes'].append(dict_instance_sub)
#PARA CUANDO HAY MAS DE UN PUNTO EN EL NOMBRE
        name = key.split('.')[0] 
        # pattern = re.compile(r'(.+?)\.jpg\d+')
        # name = pattern.match(key)
        # name = str(name.group(1))
        # print('KEY:', key)
        # print('NAME:', name)
        dict_instance['imagePath'] = name + '.jpg'
        dict_instance['imageData'] = 'test'
        dict_instance['imageHeight'] = 600
        dict_instance['imageWidth'] = 800
        
        with open(name+'.json', 'w') as fp:
            json.dump(dict_instance, fp)

def split_dataset(train_portion, json_files_list, images_src, images_dst):

    random.seed(4)
    random.shuffle(json_files_list)

    train_data = json_files_list[:round(len(json_files_list) * train_portion)]
    test_data = json_files_list[round(len(json_files_list) * train_portion):]

    print(f'Num train: {len(train_data)} Num test: {len(test_data)}')

    for train_file in train_data:
        #print(item.split('.')[1])
        #print('TRAIN_FILE:',train_file)
        #image = train_file.split('.')[0] + '.jpg'
        image = train_file.rstrip().replace("json", "jpg")
        label = train_file 


        image_src_path = os.path.join(images_src,image)
        image_dst_path = os.path.join(images_dst,image)
        image_dst_path_train = os.path.join(images_dst,'train',image)

        label_path = label
        # print('IMAGE', image)
        # print('LABEL', label)
        # print('LABEL_PATH',label_path)
        label_dst_path = os.path.join(images_dst,label)
        label_dst_path_train = os.path.join(images_dst,'train',label)
       
        try:
            shutil.copy(image_src_path, image_dst_path)
        except:
            print(f'Train 1File: {image_src_path} cannot be moved into {image_dst_path}')
        try:
            shutil.copy(image_src_path, image_dst_path_train)
        except:
            print(f'Train 2File: {image_src_path} cannot be moved int {image_dst_path_train}')
        #Copy label into general folder and also train folder
        try:
            shutil.copy(label_path, label_dst_path)
            shutil.copy(label_path, label_dst_path_train)
        #shutil.copy(name + '.json', '/content/drive/MyDrive/Banano/Dataset/etiquetado_instancias/02_40metros')
        except:
            print(f'File: {label_path} cannot be moved')
    
    for test_file in test_data:
        #print(item.split('.')[1])
        #image = test_file.split('.')[0] + '.jpg'
        image = test_file.rstrip().replace("json", "jpg")

        label = test_file 

        image_src_path = os.path.join(images_src,image)
        image_dst_path = os.path.join(images_dst,image)
        image_dst_path_test = os.path.join(images_dst,'test',image)

        label_path = label
        label_dst_path = os.path.join(images_dst,label)
        label_dst_path_test = os.path.join(images_dst,'test',label)

        #print(f'Copying image: {image} and file: {label}')
        #print(f'Copying image from: {image_src_path} to: {image_dst_path}')
        #print(f'Copying: {+".json"}')
        
        #Copy image into general folder and also train folder        
        try:
            shutil.copy(image_src_path, image_dst_path)
            shutil.copy(image_src_path, image_dst_path_test)
        except:
            print(f'File: {image_src_path} cannot be moved')
        #Copy label into general folder and also train folder
        try:
            shutil.copy(label_path, label_dst_path)
            shutil.copy(label_path, label_dst_path_test)
        #shutil.copy(name + '.json', '/content/drive/MyDrive/Banano/Dataset/etiquetado_instancias/02_40metros')
        except:
            print(f'File: {label_path} cannot be moved')


        

if __name__ == "__main__":
    print('ON Main')
    

    #Get data from images with the presence of symptoms 
    print(f'via_file: {args.via_file}')
    dict_individual_files = get_individual_files(args.via_file)
    #print(f'dict: {dict_individual_files}')
    #Generate json label file for each image 
    generate_files(dict_individual_files)

    #Get list of all json files present (Excluding the global one)
    json_files_list = glob.glob('*.json')

    #Split labels in train and test dataset

    train_portion = 0.8

    split_dataset(train_portion, json_files_list, args.images_src, args.images_dst)


    #Move images and labels to their corresponding dataset 


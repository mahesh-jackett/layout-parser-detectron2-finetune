import json
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split
from detectron2.data import DatasetCatalog, MetadataCatalog


def build_detectron_data_from_json(data:str):
    '''
    Build data in accordance with Detectron 2 from the json file
    '''
    if isinstance(data,str):
        with open(data) as f:
            data = json.load(f)
        
    image_Id2boxes_mapping = {}

    for item in data["annotations"]:
        image_id = item["image_id"]

        to_append = {'bbox': item["bbox"],
            'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': float(item['category_id'])}

        if image_id in image_Id2boxes_mapping:
            image_Id2boxes_mapping[image_id].append(to_append)
        
        else:
            image_Id2boxes_mapping[image_id] = [to_append]


    final_data = []
    for image_item in data["images"]:
        image_id = image_item["id"]
        if image_id in image_Id2boxes_mapping:
            
            to_append = {'file_name': image_item['file_name'],
            "image_id":image_id,
            'width': image_item["width"],
            'height': image_item["height"],
            'annotations': image_Id2boxes_mapping[image_id]}

            final_data.append(to_append)
    
    return train_test_split(final_data, test_size = 0.15, random_state = 42)


def dummy_register_fun(x):
    '''
    "DatasetCatalog.register()" needs a function which takes NO argument and returns a List of dictionaries. We'll wrap this function in lambda with 
    train, valid datasets which we have alreated with  "build_detectron_data_from_json()"
    '''
    return x


def build_register_detectron_data(json_file_or_path, Data_Resister_training, Data_Resister_valid, thing_classes):
    """
    Build and register Data from json file or dict. Register wit the names of the datasets you want to use
    """
    train, val = build_detectron_data_from_json(json_file_or_path)

    DatasetCatalog.register(Data_Resister_training,lambda: dummy_register_fun(train))
    MetadataCatalog.get(Data_Resister_training).set(thing_classes = thing_classes)
    
    DatasetCatalog.register(Data_Resister_valid,lambda: dummy_register_fun(val))
    MetadataCatalog.get(Data_Resister_valid).set(thing_classes = thing_classes)
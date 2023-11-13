import argparse
import os
import os.path as osp

import mmcv
import mmengine
import xml.etree.ElementTree as ET
IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg')


def check_existence(file_path: str):
    """Check if target file is existed."""
    if not osp.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')


def get_image_info(image_dir, idx, file_name):
    """Retrieve image information."""
    img_path = osp.join(image_dir, file_name)
    check_existence(img_path)

    img = mmcv.imread(img_path)
    height, width = img.shape[:2]
    img_info_dict = {
        'file_name': file_name,
        'id': idx,
        'width': width,
        'height': height
    }
    return img_info_dict, height, width


def convert_bbox_info(label, idx, obj_count, image_height, image_width):
    """Convert yolo-style bbox info to the coco format."""
    label = label.strip().split()
    x1 = float(label[0].split(',')[0])
    y1 = float(label[0].split(',')[1])
    x2 = float(label[2].split(',')[0])
    y2 = float(label[2].split(',')[1])

    cls_id = 0
    width = max(0., x2 - x1)
    height = max(0., y2 - y1)
    coco_format_info = {
        'image_id': idx,
        'id': obj_count,
        'category_id': cls_id,
        'bbox': [x1, y1, width, height],
        'area': width * height,
        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
        'iscrowd': 0
    }
    obj_count += 1
    return coco_format_info, obj_count


def organize_by_existing_files(image_dir: str, existed_categories: list):
    """Format annotations by existing train/val/test files."""
    categories = ['train', 'test']
    image_list = []

    for cat in categories:
        if cat in existed_categories:
            txt_file = osp.join(image_dir, f'{cat}.txt')
            print(f'Start to read {cat} dataset definition')
            assert osp.exists(txt_file)

            with open(txt_file) as f:
                img_paths = f.readlines()
                img_paths = [
                    os.path.split(img_path.strip())[1]
                    for img_path in img_paths
                ]  # split the absolute path
                image_list.append(img_paths)
        else:
            image_list.append([])
    return image_list[0], image_list[1], image_list[2]

def convert(split,classes,subset_img_dir,label_dir,out_file):
    indices = os.listdir(subset_img_dir)
    total = len(indices)

    dataset = {'images': [], 'annotations': [], 'categories': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls})

    obj_count = 0
    skipped = 0
    converted = 0
    for idx, image in enumerate(mmengine.track_iter_progress(indices)):
        img_info_dict, image_height, image_width = get_image_info(
            subset_img_dir, idx, image)

        dataset['images'].append(img_info_dict)

        img_name = osp.splitext(image)[0]
        label_path = f'{osp.join(osp.join(label_dir, split),img_name)}.xml'
        if not osp.exists(label_path):
            # if current image is not annotated or the annotation file failed
            print(
                f'WARNING: {label_path} does not exist. Please check the file.'
            )
            skipped += 1
            continue
        tree = ET.parse(label_path)
        root = tree.getroot()
        for child in root:
            if child.tag == 'table':
                coords = child.find('Coords').attrib['points']
                coco_info, obj_count = convert_bbox_info(
                coords, idx, obj_count, image_height, image_width)
                dataset['annotations'].append(coco_info)
        converted += 1


    print(f'Saving converted results to {out_file} ...')
    mmengine.dump(dataset, out_file)

    # simple statistics
    print(f'Number of images found: {total}, converted: {converted},',
          f'and skipped: {skipped}. Total annotation count: {obj_count}.')
    print('You can use tools/analysis_tools/browse_coco_json.py to visualize!')


def convert_yolo_to_coco(root_dir: str):
    print(f'Start to load existing images and annotations from {root_dir}')
    check_existence(root_dir)

    # check local environment
    label_dir = osp.join(root_dir, 'Annotations')
    image_dir = osp.join(root_dir, 'Images')
    check_existence(label_dir)
    check_existence(image_dir)

    print(f'All necessary files are located at {image_dir}')

    existed_categories = []

    # prepare the output folders
    output_folder = osp.join(root_dir, 'annotations')
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
        check_existence(output_folder)

    classes = ['table']
    
    # train
    split = 'TrainSet'
    subset_img_dir = osp.join(image_dir,split)
    out_file = osp.join(output_folder, 'trainval.json')
    convert(split, classes, subset_img_dir,label_dir,out_file)

    # test
    split = 'TestSet'
    subset_img_dir = osp.join(image_dir,split)
    out_file = osp.join(output_folder, 'test.json')
    convert(split, classes, subset_img_dir,label_dir,out_file)


if __name__ == '__main__':
    root = "/mnt/data/ICDAR2017_POD_dataset_supplement-main/"
    convert_yolo_to_coco(root)

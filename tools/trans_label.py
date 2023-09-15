import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "/mnt/data0/sweeper/merge/source_data"

# set export dir
export_dir = "/mnt/data0/sweeper/merge/annotations/"

# set train split rate
train_split_rate = 0.85

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir, train_split_rate)
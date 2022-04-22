import splitfolders

img_dir = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Project2.0\xray_images'
new_img_dir = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Project2.0\splitted_data'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(img_dir, output=new_img_dir, seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values

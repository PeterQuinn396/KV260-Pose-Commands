import glob
import pickle
import os
from PIL import Image
import cv2.cv2
import numpy as np

CATEGORIES = ['up', 'down', 'left', 'right', 'fist', 'palm']
MAX_IMAGES = 50
TEST_SIZE = 10

# CATEGORIES = ['up']
# MAX_IMAGES = 1
# TEST_SIZE = 0


def make_pickle_file(base_dir: str = 'data'):
    train_x, train_y = [], []
    test_x, test_y = [], []
    for index, category in enumerate(CATEGORIES):
        path = os.path.join(base_dir, category)
        jpgs = sorted(glob.glob(f'{path}/*.jpg'))
        # print(f'Generating train set for {category}...')
        # for i in range(MAX_IMAGES - TEST_SIZE):
        #     im = Image.open(jpgs[i])
        #     arr = np.array(im)
        #     arr = np.moveaxis(arr, -1, 0)  # move channels to front for pytorch
        #     train_x.append(arr)
        #     train_y.append(index)

        print(f'Generating test set for {category}...')
        for i in range(MAX_IMAGES - TEST_SIZE, MAX_IMAGES):
            im = Image.open(jpgs[i])
            arr = np.array(im)
            arr = np.moveaxis(arr, -1, 0)  # move channels to front for pytorch
            test_x.append(arr)
            test_y.append(index)

    # train = {
    #     'train_x': train_x,
    #     'train_y': train_y,
    # }

    test = {
        'test_x' : test_x,
        'test_y' : test_y,
    }
    #print(output)
    print("Writing pickle file...")
    # with open('custom_dataset_train.pckl', 'wb+') as f:
    #     pickle.dump(train, f)
    #
    with open('custom_dataset_test.pckl', 'wb+') as f:
        pickle.dump(test, f)

if __name__ == '__main__':
    make_pickle_file()

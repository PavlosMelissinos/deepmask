"""Docstring placeholder"""
import numpy as np
import cv2
import sys
import json
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

FILENAME = sys.argv[1]
ANNFILE = 'data/annotations/instances_train2014.json'


def png_to_json(imfile):
    """This function does something"""
    img = cv2.imread(imfile, cv2.IMREAD_UNCHANGED)
    width = img.shape[0]
    height = img.shape[1]
    dims = width * height
    depth = img.shape[2]

    img = np.reshape(img, (dims, depth))
    print img.shape


def load_dataset(annotation_file):
    """This function does something"""
    dataset = json.load(open(annotation_file, 'r'))
    return dataset


def load_image(img):
    """This function does something"""
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_coco(ann_file):
    """This function does something"""
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=['person', 'dog'])
    # img_ids = coco.getImgIds(catIds=cat_ids)
    img_id = 428412
    # img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    img = coco.loadImgs(img_id)

    print img
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    ann_ids = coco.getAnnIds(imgIds=428412, catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    
    # print anns[0]

    # print json.dumps(anns, indent=4, sort_keys=True)

    parsed = anns[0]['segmentation']
    # print type(parsed), len(parsed), len(parsed[0])
    # parsed = json.loads(anns)
    # print 'parsed all\n', json.dumps(parsed[0])

    # print 'segmentation \n', anns['segmentation']
    coco.showAnns(anns)
    # for ann in COCO.annotations:
    #     print ann
    #     raw_input("Pre")


def load_coco_img(img_id):
    """This function does something"""
    img = io.imread('http://mscoco.org/images/%d'%(img_id))
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def split_dataset(ann_file):
    with open(ann_file) as infile:
        o = json.load(infile)
        print type(o), list(o.keys())
        print type(o['info']), list(o['info'].keys())
        print type(o['images']), len(o['images'])
        print type(o['licenses']), len(o['licenses'])
        print type(o['annotations']), len(o['annotations'])
        print type(o['categories']), len(o['categories'])
        print o['annotations'][0]
        print o['images'][0]
        # o = o['segmentation']
        # print o.keys(), type(o)
        # chunk_size = 1
        # for i in xrange(0, len(o), chunk_size):
        #     with open('file_' + str(i // chunk_size) + '.json', 'w') as outfile:
        #         json.dump(o[i:i + chunk_size], outfile)
        #         print 'dumped a chunk'

# png_to_json(filename)
# load_coco(ANNFILE)
# load_coco_img(428412)
# data = load_dataset(ANNFILE)
# print 'test'
split_dataset(ANNFILE)
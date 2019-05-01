import os
# import cv2
import xml.etree.ElementTree as etxml
import random
from sklearn.utils import shuffle
import numpy as np
import skimage.io
import skimage.transform
import glob

class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    # self._categories = categories
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  # @property
  # def categories(self):
  #   return self._categories

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` samples ."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # update this for each epoch
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      # print(batch_size,self._num_examples)
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

'''
Used for classification
'''
def read_dataset(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls, _ = load_dataset(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets


'''
Used for classification
'''
def load_dataset(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    categories=[]

    print('=======Ready to load the pictures======')
    for fields in classes:   
        categories.append(fields)
        index = classes.index(fields)
        print('Now ready to read [{}] category (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        #======= calculat the number of files=========
        path1 = os.path.join(train_path, fields)
        print(path1)
        print ("the number of ", fields, "is ", len([name for name in os.listdir(path1) if os.path.isfile(os.path.join(path1, name))]))
        #======= calculat the number of files=========
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls, categories
  
'''
Used for object detection
'''
def get_data(batch_size,img_path,xml_path):
    # def get_ground_truth(xml_path):
    def get_actual_data_from_xml(xml_file):
        actual_item = []
        try:
            annotation_node = etxml.parse(xml_file).getroot()
            # print("now enter try")
            # print(annotation_node)
            img_width =  float(annotation_node.find('size').find('width').text.strip())
            # print(img_width)
            img_height = float(annotation_node.find('size').find('height').text.strip())
            object_node_list = annotation_node.findall('object')       
            # print(object_node_list)
            for obj_node in object_node_list:                       
                # print(obj_node)
                lable = lable_arr.index(obj_node.find('name').text.strip())
                bndbox = obj_node.find('bndbox')
                x_min = float(bndbox.find('xmin').text.strip())
                # print(x_min)
                y_min = float(bndbox.find('ymin').text.strip())
                x_max = float(bndbox.find('xmax').text.strip())
                y_max = float(bndbox.find('ymax').text.strip())
                # 位置数据用比例来表示，格式[center_x,center_y,width,height,lable]
                actual_item.append([((x_min + x_max)/2/img_width), ((y_min + y_max)/2/img_height), ((x_max - x_min) / img_width), ((y_max - y_min) / img_height), lable])
            return actual_item  
        except:
            return None
        
    train_data = []
    actual_data = []
    lable_arr = ['bully','victim']
    # 图像白化，格式:[R,G,B]
    whitened_RGB_mean = [123.68, 116.78, 103.94]
    
    # print(file_name_list)
    file_name_list = [os.path.basename(x) for x in glob.glob('/Users/tarus/OnlyInMac/bully_data/bully_merge_train/JPEGImages/*.jpg')]
    # print(file_name_list)
    file_list = random.sample(file_name_list, batch_size)
    f = open("./debug.txt", 'w+') 
    for f_name in file_list :
        print(f_name)
        #img_path = './train_datasets/voc2007/JPEGImages/' + f_name
        # img_path = '/Users/tarus/OnlyInMac/bully_data/bully_test/JPEGImages/' + f_name
        # img_path = '/Users/tarus/OnlyInMac/dilated_cnn/VOC2012/JPEGImages/' + f_name
        # img_path = '/Users/tarus/OnlyInMac/dilated_cnn/VOC2007/JPEGImages/' + f_name
        img_file= img_path + f_name
        # print(img_file)
        # img_path = '/Users/tarus/OnlyInMac/bully_data/bully_merge/JPEGImages/' + f_name
        #xml_path = './train_datasets/voc2007/Annotations/' + f_name.replace('.jpg','.xml')
        # xml_path = '/Users/tarus/OnlyInMac/bully_data/bully_test/Annotations/' + f_name.replace('.jpg','.xml')
        xml_file= xml_path + f_name.replace('.jpg','.xml')
        # xml_path = '/Users/tarus/OnlyInMac/bully_data/bully_merge/Annotations/' + f_name.replace('.jpg','.xml')
        # xml_path = '/Users/tarus/OnlyInMac/dilated_cnn/VOC2007/Annotations/' + f_name.replace('.jpg','.xml')
        # xml_path = '/Users/tarus/OnlyInMac/dilated_cnn/VOC2012/Annotations/' + f_name.replace('.jpg','.xml')
        # if os.path.splitext(img_path)[1].lower() == '.jpg' :
        # print(os.path.splitext(img_file)[1].lower())
        if os.path.splitext(img_file)[1].lower() == '.jpg' :
            actual_item = get_actual_data_from_xml(xml_file)
            # print(actual_item)
            if actual_item != None :
                actual_data.append(actual_item)
            else :
                print('Error : '+xml_file)
                continue
            img = skimage.io.imread(img_file)
            # print("img path is")
            # print(img_file)
            img = skimage.transform.resize(img, (300, 300))
            # print("img value is")
            # print(img)
            # print(img,file=f)
            # 图像白化预处理
            img = img - whitened_RGB_mean
            train_data.append(img)
            
    return train_data, actual_data,file_list
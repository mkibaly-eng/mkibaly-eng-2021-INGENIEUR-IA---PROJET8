from flask import Flask, render_template
from flask import request
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
from tensorflow import keras
import segmentation_models as sm
sm.set_framework('tf.keras')

app = Flask(__name__)
app.config["DEBUG"] = True

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        j = i + 1
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
#        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.imshow(display_list[i])
        plt.savefig("Users/loubna1026/SegApp/static/img/img_%s.png" % j, format='png')
        plt.axis('off')
    plt.show()


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)


CFG = {
    "data": {
        "path": "oxford_iiit_pet:3.*.*",
        "image_size": 256,
        "load_with_info": True
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epoches": 20,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}



class UnetInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.image_size = self.config.data.image_size
        self.model = tf.keras.models.load_model('Users/loubna1026/SegApp/unet.h5',
                     custom_objects={'focal_loss_plus_jaccard_loss': sm.losses.categorical_focal_jaccard_loss,
                                     'iou_score': sm.metrics.iou_score})

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape= tensor_image.shape
        tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
        pred = self.model.predict(tensor_image)
        pred = pred[0]
        pred_mask = tf.argmax(pred, axis=-1)
        display([tensor_image[0], pred_mask])
        return render_template('image.html')
#        pred = pred.numpy().tolist()
#        return {'segmentation_output':pred}


u_net = UnetInferrer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_img = request.form['name']
#       return 'Identifiant image %s' % (text_img)
        image = np.asarray(Image.open('Users/loubna1026/SegApp/data/test/image/%s' % text_img)).astype(np.float32)
        return u_net.infer(image)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)


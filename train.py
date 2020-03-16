import os

import numpy as np
import pandas as pd
from PIL import Image

from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.python.keras.applications import vgg16
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img

imgs_path = "../input/dekapictures"
imgs_width, imgs_height = 224, 224
# model_path = ''

nb_closest_images = 5

def load_trained_model():
    vgg_model = vgg16.VGG16(weights='imagenet')
    feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)
    feature_extractor.summary()
    return feature_extractor

def get_images_paths():
    files = [imgs_path + '/' + x for x in os.listdir(imgs_path) if 'jpg' in x]
    print('number of images:', len(files))
    return files

def test_one_image(files):
    original = load_img(files[0], target_size=(imgs_width, imgs_height))
    plt.imshow(original)
    plt.show()
    print('image successfully loaded')
    return original

def extract_one_image_feature(image):
    numpy_image = img_to_array(image)
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('image batch size', image_batch.shape)
    preprocessed_image = preprocess_input(image_batch.copy())
    return preprocessed_image

def get_extracted_features(image, extractor):
    img_features = extractor.predict(image)
    print('features successfully extracted!')
    return img_features

def feed_batch_image(files):
    new_images = []

    for f in files:
        filename = f
        original_image = load_img(filename, target_size=(imgs_width, imgs_height))
        numpy_image = img_to_array(original_image)
        image_batch = np.expand_dims(numpy_image, axis=0)
        new_images.append(image_batch)
    
    images = np.vstack(new_images)
    processed_imgs = preprocess_input(images.copy())
    return processed_imgs

def extract_features(extractor, processed_images):
    imgs_features = extractor.predict(processed_images)
    print('features succesfully extracted!')
    print('number of image features: ', imgs_features.size)
    return imgs_features

def create_sim_df(imgs_features, files):
    cosine_sim = cosine_similarity(imgs_features)
    cosine_sim_df = pd.DataFrame(cosine_sim, columns=files, index=files)
    return cosine_sim_df

def retrieve_similar_images(given_image, cosine_similarity_df):
    print('original image:')
    original = load_img(given_image, target_size=(imgs_width, imgs_height))
    plt.imshow(original)
    plt.show()
    
    print('most similar products........')

    closest_imgs = cosine_similarity_df[given_image].sort_values(ascending=False)[1:nb_closest_images + 1].index
    closest_imgs_scores = cosine_similarity_df[given_image].sort_values(ascending=False)[1:nb_closest_images + 1]

    for i in range(0, len(closest_imgs)):
        original = load_img(closest_imgs[i], target_size=(imgs_width, imgs_height))
        plt.imshow(original)
        plt.show()
        print('similarity score:', closest_imgs_scores[i])


def process_images():
    model = load_trained_model()
    images = get_images_paths()
    processed_imgs = feed_batch_image(images)
    extracted_features = extract_features(model, processed_imgs)
    similarity_df = create_sim_df(extracted_features, images)
    return similarity_df, images

if __name__ == '__main__':
    sim_df, images = process_images()
#     print(sim_df, images)
    retrieve_similar_images(images[0], sim_df)
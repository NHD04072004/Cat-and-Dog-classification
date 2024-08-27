import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from arguments import get_args

def predict(img_path, model_path, img_size):
    model = tf.keras.models.load_model(model_path)

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return 'dog'
    else:
        return 'cat'

if __name__ == '__main__':
    args = get_args()
    img_path = input("Enter the path to the image: ")
    result = predict(img_path, args.model_save_path, args.img_size)
    print(f'The image is a {result}.')

from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('tomato_disease2.h5')

# Define the class names
class_names = ['Early_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'healthy', 'powdery_mildew']

def predict(model, img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file_path = "./static/" + file.filename
        file.save(file_path)
        
        predicted_class, confidence = predict(model, file_path)
        
        return render_template('index.html', prediction=predicted_class, confidence=confidence, user_image=file_path)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

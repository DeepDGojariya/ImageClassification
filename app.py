from flask import Flask,render_template,request
import numpy as np
import tensorflow 
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = tensorflow.keras.models.load_model('model.h5')
opDict = {0:'Buildings',1:'Forests',2:'Glaciers',3:'Mountains',4:'Sea',5:'Street'}

def predict_class(img_path):
    test_image = image.load_img(img_path, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    result=list(result[0])
    img_index = result.index(max(result))
    return opDict[img_index]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def submit():
    if request.method=='POST':
        user_file = request.files['user_img']
        path = './static/{}'.format(user_file.filename)
        user_file.save(path)
        output = predict_class(path)
        context = {
            'image':path,
            'prediction':output
        }
        return render_template('index.html',context=context)



if __name__ == '__main__':
    app.run(debug=True)
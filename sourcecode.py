from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
from tensorflow.keras.models import save_model
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pyttsx3
import cv2
def classifier_123(img):
    save_path = r"C:\Users\HP\Documents\project"
    def build_finetune_model(save_path,base_model, dropout, fc_layers, num_classes):
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Flatten()(x)
        for fc in fc_layers:
            x = Dense(fc, activation='relu')(x)
            x = Dropout(dropout)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        finetune_model = Model(inputs=base_model.input, outputs=predictions)
        save_model(finetune_model, save_path)
    class_list = ['Real', 'Fake']
    FC_Layers = [1024, 1024]
    dropout = 0.5
    height = 300
    width = 300
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, 3))
    finetune_model = build_finetune_model(save_path,base_model, dropout=dropout, fc_layers=FC_Layers, num_classes=len(class_list))
    #ing = image.load_img(r'C:\Users\HP\Documents\data set\training' , target_size=(300, 300))
    img = cv2.resize(img, (300, 300))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    finetune_model.load_weights(r"C:\Users\HP\Documents\project\saved_model.pb")
    output = finetune_model.predict(img)
    if output[0][0] > output[0][1]:
        k = 'Fake'
        print("Fake")
    else:
        k = 'Real'
        print("Real")
    text = ('You had', k, 'Note')
    speak(text)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# Load the pre-trained model
model = load_model(r"C:\Users\HP\Documents\project\cnn_model_Classifier.h5", compile=False)

# Initialize webcam
cam = cv2.VideoCapture(0)

for a in range(0, 50):
    ret, img = cam.read()
    if ret:
        file = r'C:\Users\HP\Documents\campic\im' + str(a) + '.jpg'
        cv2.imwrite(file, img)
        img1 = img
        cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        print("Image" + str(a) + "saved")

cam.release()
cv2.destroyAllWindows()

# Predict and classify images
path = os.path.join(r'C:\Users\HP\Documents\campic')
files = os.listdir(path)
print(files)
test_image = random.choice(files)
print('the test is', test_image)
test_image = os.path.join(path, test_image)
test_image = image.load_img(test_image, target_size=(224, 224))
print(test_image)
test_image = image.img_to_array(test_image)
test_image = test_image / 255
test_image = np.expand_dims(test_image, axis=0)
k = ''
frame1 = model.predict(test_image)
a = np.argmax(frame1)
print(a)
if a == 0:
    k = '10'
    print("10")
elif (a == 1):
    k = '100'
    print("100")
elif (a == 2):
    k = '20'
    print("20")
elif (a == 3):
    k = '200'
    print("200")
elif (a == 4):
    k = '2000'
    print("2000")
elif (a == 5):
    k = '50'
    print("50")
elif (a == 6):
    k = '500'
    print("500")
elif (a == 7):
    print("not a note")

text = ('You had', k, 'Note')
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()
engine.stop()
classifier_123(img=img1)

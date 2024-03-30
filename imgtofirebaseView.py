import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
import base64

# Set up Firebase credentials
try:
    cred = credentials.Certificate('/home/gits/Desktop/40000-Aarakshak-model/aarakshak-ai-cam-firebase-adminsdk-14u0q-96da7fcdbe.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://aarakshak-ai-cam-default-rtdb.firebaseio.com//',
        # 'storageBucket': 'aarakshak-web-detect.appspot.com/' # Replace with your Firebase Storage bucket URL
    })
except Exception as e:
    print("Error initializing Firebase:", e)
    exit()

modelpath ='detect.tflite'
lblpath ='labelmap.txt'
min_conf = 0.55

interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

try:
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print("Error reading labelmap file:", e)
    exit()

cap = cv2.VideoCapture(0)

# Initialize Firebase Storage
try:
    bucket = storage.bucket('aarakshak-ai-cam.appspot.com')
    # Or if you want to specify a specific bucket, use:
    # bucket = storage.bucket('videodetection')  # Replace with your Firebase Storage bucket name
except Exception as e:
    print("Error initializing Firebase Storage bucket:", e)
    exit()

def send_to_firebase(object_name, image_data):
    try:
        # Upload image to Firebase Storage
        current_date_time = datetime.now()
        formatted_date_time = current_date_time.strftime("%d-%m-%Y")
        formatted_date_time1 = current_date_time.strftime("%H-%M-%S")
        image_filename = f"triger_{formatted_date_time}_{formatted_date_time1}.jpg"
        # image_filename = f"triger_{formatted_date_time1}.jpg"

        # Decode the base64-encoded image data
        decoded_image = base64.b64decode(image_data)

        # Upload the image data to a storage location
        blob = bucket.blob(f"triger/{image_filename}")
        blob.upload_from_string(decoded_image, content_type='image/jpeg')

        # Get the public URL of the uploaded image
        image_url = blob.public_url

        # Send object details along with the image URL to Firebase Realtime Database
        ref = db.reference('/Detection_Results')
        ref.set({
            'Class Label': object_name,
            'Location': 'https://maps.app.goo.gl/ht7B1svjBy8kqjhv9',
            'Date': formatted_date_time,
            'Time': formatted_date_time1,
            'ImageURL': image_url
        })
    except Exception as e:
        print("Error sending data to Firebase:", e)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Couldn't read frame from RTSP stream.")
        break

    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]   # Confidence of detected objects

        for i in range(len(scores)):
            if (scores[i] > min_conf) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                current_date_time = datetime.now()
                formatted_date_time = current_date_time.strftime("%d-%m-%Y")
                formatted_date_time1 = current_date_time.strftime("%H-%M-%S")
                print(object_name)
                print(formatted_date_time)
                print(formatted_date_time1)
                print("At Poornima College of Engineering")

                # Convert the detected frame to base64 format
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = base64.b64encode(buffer).decode('utf-8')

                # Send object_name and image to Firebase
                send_to_firebase(object_name, image_data)

    except Exception as e:
        print("Error processing frame:", e)

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

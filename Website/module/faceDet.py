from deepface import DeepFace
import os

# example of face detection with mtcnn
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from os import listdir
from os.path import isfile, join


# extract a single face from a given photograph
def save_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# pixels = pixels.resize(required_size)
	# create the detector, using default weights
	detector = MTCNN()
	# breakpoint()
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	image.save('cropped_face.jpg')
	return os.path.join(os.getcwd()+'/cropped_face.jpg')

# load the photo and extract the face
# pixels = extract_face('sharon_stone1.jpg')
# plot the extracted face
# pyplot.imshow(pixels)
# show the plot
pyplot.show()

DB_FOLDER = os.path.join(os.getcwd()+'/module/faces')

def get_face(image_path):
    face_path = save_face(image_path)
    # breakpoint()
    # for file in listdir(DB_FOLDER):
    #       if isfile(join(DB_FOLDER, file)) and file[-3:]=="jpg":
    #               verification = DeepFace.verify(img1_path = face_path, img2_path = join(DB_FOLDER, file), enforce_detection=False)
    #               print(verification)
                

    # for file in listdir(DB_FOLDER):
	#     if isfile(join(DB_FOLDER, f)):

    print(face_path)
    return DeepFace.find(img_path=face_path, db_path = DB_FOLDER,enforce_detection=False, model_name="Facenet512")
	# return DeepFace.stream(db_path = “C:/facial_db”)


# import os

# [                                            identity  target_x  target_y  target_w  target_h  source_x  source_y  source_w  source_h  VGG-Face_cosine
# 0  /home/shankar-pt-7332/Documents/PersonalProjec...         0         0       224       224         0         0       224       224         0.336048]

# [                                            identity  target_x  target_y  target_w  target_h  source_x  source_y  source_w  source_h  VGG-Face_cosine
# 0  /home/shankar-pt-7332/Documents/PersonalProjec...         0         0       224       224         0         0       224       224         0.372879]

# {'verified': True, 'distance': 0.3360481340792366, 'threshold': 0.4, 'model': 'VGG-Face', 'detector_backend': 'opencv', 'similarity_metric': 'cosine', 'facial_areas': {'img1': {'x': 0, 'y': 0, 'w': 224, 'h': 224}, 'img2': {'x': 0, 'y': 0, 'w': 224, 'h': 224}}, 'time': 1.9}
# /home/shankar-pt-7332/Downloads/Ajith_kumar_2019.jpg
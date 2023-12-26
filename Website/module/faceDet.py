from deepface import DeepFace

dfs = DeepFace.find(img_path = "C:\\Users\\Shankaranaarayanan\\Desktop\\Shankar\\KCT\\4thyear\\Final yr Project\\Code\\VisionSync\\Website\\module\\faces\\passport_photo.png", db_path = "C:\\Users\\Shankaranaarayanan\\Desktop\\Shankar\\KCT\\4thyear\\Final yr Project\\Code\\VisionSync\\Website\\module\\faces")

print(dfs)
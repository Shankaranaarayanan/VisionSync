
import cv2

img = cv2.imread('uploads\\billboard.jpg',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# inverse binary
th = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# find contours and sort them from left to right
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=lambda x: [cv2.boundingRect(x)[0], cv2.boundingRect(x)[1]])

#initialize dictionary
board_dictionary = {}

# iterate each contour and crop bounding box
for i, c in enumerate(contours):
    x,y,w,h = cv2.boundingRect(c)
    crop_img = img[y:y+h, x:x+w]

    # feed cropped image to easyOCR module
    # results = reader.readtext(crop_img)
    cv2.imshow('img',crop_img)
    cv2.waitKey

    # result is output per line
    # create a list to append all lines in cropped image to it
    # board_text = []
    # for (bbox, text, prob) in results:
    #   board_text.append(text)
    
    # # convert list of words to single string
    # board_para = ' '.join(board_text)
    # #print(board_para)

    # # store string within a dictionary
    # board_dictionary[str(i)] = board_para
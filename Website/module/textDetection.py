from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import os



ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory



def get_result(imagepath, resultpath):
    result = ocr.ocr(imagepath, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    result = result[0]
    image = Image.open(imagepath).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    font_path = os.path.normpath(os.getcwd() + os.sep + os.pardir +'\\Fonts\\simfang.ttf') 
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)

    im_show.save(resultpath)



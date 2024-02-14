import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
from module.textDetection import get_text
from module.faceDet import get_face
from module.questionAnsWithLxmrt import get_result

UPLOAD_FOLDER = os.path.join(os.getcwd()+'/uploads')
STATIC_FOLDER = 'static'
resultpath = os.path.join(STATIC_FOLDER, 'result.jpg')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','avif','jpg_large'}



app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '12345'


file_name = {

}

questions = []
responses = []

face_det_results = {

}




def get_results(input,output, option):
    print(input,output)
    if option=='text':
        return get_text(input, output)
    elif option=='faceDet':
        res = get_face(input)
        face_det_results[input] = res
        return




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        pass
    return render_template('home.html')


@app.route('/getQuestion', methods=['GET', 'POST'])
def getQuestion():
    if request.method == 'POST':
        question = request.form['question']
        print(file_name)
        breakpoint()
        img_path = file_name['interactive']
        response = get_result(img_path, question)
        questions.append(question)
        responses.append(response)
        return render_template('getquestion.html', qns = questions, res = responses, len = len(questions))
        
    return render_template('getquestion.html')


@app.route('/uploadImg/<option>', methods=['GET', 'POST'])
def upload_file(option):
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_name[option] = filepath
            if option=='interactive':
                return redirect(url_for('getQuestion'))
            get_results(filepath, resultpath, option)
            return redirect(url_for('put_results'))
        else:
            flash('Unsuported file!!')
            
            
    return render_template('index.html')


@app.route('/results', methods=['GET','POST'])
def put_results():
    if request.method == "POST":
        return redirect(url_for('home_page'))
    print(face_det_results)
    if face_det_results == {}:
        return render_template('text_res.html', result = resultpath)
    else:
        for i in face_det_results:
            print(i)
            copy = face_det_results[i]
        faces = []
        face_det_results.clear()
        copy = copy[0]
        if (not copy.empty):
            for i in range(len(copy)):
                print(copy['VGG-Face_cosine'][i])
                if float(copy['VGG-Face_cosine'][i])<0.35:
                    face = copy['identity'][0].split('/')[-1]
                    faces.append(face[0:len(face)-4])
        return render_template('face_res.html', result = faces)

if __name__=='main':
    app.run(debug=True)
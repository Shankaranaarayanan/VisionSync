import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
from module.textDetection import get_result

UPLOAD_FOLDER = os.path.join(os.getcwd()+'\\uploads')
STATIC_FOLDER = 'static'
resultpath = os.path.join(STATIC_FOLDER, 'result.jpg')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}



app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '12345'



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
            get_result(filepath, resultpath)
            return redirect(url_for('get_results'))
        else:
            flash('Unsuported file!!')
            
            
    return render_template('index.html')


@app.route('/results', methods=['GET','POST'])
def get_results():
    if request.method == "POST":
        return redirect(url_for('upload_file'))
    return render_template('result.html', result = resultpath)

if __name__=='main':
    app.run(debug=True)
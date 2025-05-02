from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
from cartoonGAN.transform import transform_image, transform_video
from white_box_cartoonization.cartoonize import cartoonize_image
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load White Box Cartoonization model

def allowed_file(filename, extensions=['png', 'jpg', 'jpeg', 'mp4']):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

from white_box_cartoonization.cartoonize import WB_Cartoonize, cartoonize_image

# ...

# Instantiate White Box Cartoonization model
wbc_model = WB_Cartoonize(os.path.abspath('white_box_cartoonization/saved_models'), gpu=False)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    original_image = None
    cartoon_gan_result = None
    white_box_result = None

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # CartoonGAN transformation
            cartoon_gan_output_path = os.path.join(UPLOAD_FOLDER, 'cartoon_gan_' + filename)
            if filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']:
                transform_image(filepath, cartoon_gan_output_path)
            else:
                transform_video(filepath, cartoon_gan_output_path)

            # White Box Cartoonization transformation
            white_box_output_path = os.path.join(UPLOAD_FOLDER, 'white_box_' + filename)
            if filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']:
                cartoonize_image(filepath, white_box_output_path)
            else:
                final_name = wbc_model.process_video(filepath, '30')  # assume 30 fps
                os.rename(final_name, white_box_output_path)

            original_image = filename
            cartoon_gan_result = 'cartoon_gan_' + filename
            white_box_result = 'white_box_' + filename

    return render_template('index.html', original_image=original_image, white_box_result=white_box_result)

if __name__ == '__main__':
    app.run(debug=True)

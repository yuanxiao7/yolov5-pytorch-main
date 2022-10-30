from flask import render_template

from flask_uploads import UploadSet, IMAGES, configure_uploads, patch_request_class
from flask import Flask, request
from yolov5_detector import detect
import json

app = Flask(__name__)

app.config["UPLOADED_PHOTOS_DEST"] = 'uploads'

photo = UploadSet('photos', IMAGES)
configure_uploads(app, photo)
patch_request_class(app)


# 定义总路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detector', methods=['POST'])
def detector():
    filename = photo.save(request.files['file']) #保存图片
    file_url = photo.url(filename) # 获取url
    print('1  ', file_url)
    path = photo.path(filename) # 获取存储路径
    result = detect(path)
    print('2  ', result)
    result_url = photo.url(result + '.jpg')  # 获取url
    print('3  ', result_url)
    data = {'file_url': file_url, 'result_url': result_url}
    data = json.dumps(data)
    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



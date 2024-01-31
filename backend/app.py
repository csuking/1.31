from flask import Flask, request, jsonify, send_file
from train_model import Model, SimpleModel
import os

app = Flask(__name__)
model = SimpleModel()
# 设置允许的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    return send_file(r'C:\Users\86183\Desktop\face_recognition_pyqt\frontend\front.html')

# 检查文件类型是否合法
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/recognize', methods=['POST'])
def recognize():
    # 检查是否收到了文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        # 调用模型的新预测函数
        result_index, confidence = model.predict_from_image(file)

        # 构建响应数据
        response_data = {
            'result_index': result_index,
            'confidence': confidence
        }

        # 返回JSON格式的响应
        return jsonify(response_data)

    except Exception as e:
        # 发生异常时返回错误信息
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)

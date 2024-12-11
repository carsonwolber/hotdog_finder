import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app  = Flask(__name__)
CORS(app)

@app.route('/check-hotdog', methods=['POST'])
def check_for_dog():
  file = request.files['image']
  image_data = file.read()
  image_b64 = base64.b64encode(image_data).decode('utf-8')
  return jsonify({
    'image': image_b64
  })

 
def create_app():
  return app

if __name__ == '__main__':
  app = create_app()
  with app.app_context():
    app.run(port=3000)

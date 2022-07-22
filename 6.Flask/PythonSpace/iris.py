from flask import Flask, jsonify, render_template, request
import joblib

app = Flask(__name__)
# Flask는 spring 과 비슷
@app.route("/iris")
def iris():
    sepalLength = float(request.args.get('sepalLength')) # request는 문자타입이므로 숫자로 변환
    sepalWidth = float(request.args.get('sepalWidth'))
    petalLength = float(request.args.get('petalLength'))
    petalWidth = float(request.args.get('petalWidth'))

    clf = joblib.load("./rf_iris.h5") 
    # flask는 터미널을 사용하고 .을 쓰기 위해서 터미널 경로를 맞춰준다.
    pre = clf.predict([[sepalLength, sepalWidth ,petalLength, petalWidth]])
    # 예측하기

    print(pre)
    # flutter 에 json으로 넘겨줌pws
    return jsonify({'result' : pre[0][5:]})

# render_template : html 실행
def home():
    return render_template('home.html') 

if __name__ == "__main__":
    app.run(host='127.0.0.1' , port=5000 , debug = True)

# 오른쪽 버튼 클릭 run python on terminal 클릭 

# cd PythonSpace 
# ls
# pip install flask
# pwd
# "run python on terminal"
# -> Python Terminal
# quit ctrl-c
# cd PythonSpace
# "run python on terminal"

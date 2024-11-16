from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)

parkinsons_model = pickle.load(open("parkinsons_model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html")  

@flask_app.route("/predict_page", methods=["GET"])
def predict_page():
    return render_template("predict.html") 


@flask_app.route("/predict", methods=["POST"])
def predict():
    result = None
    if request.method == 'POST':
       
        user_input = [
            request.form.get('fo'),
            request.form.get('fhi'),
            request.form.get('flo'),
            request.form.get('Jitter_percent'),
            request.form.get('Jitter_Abs'),
            request.form.get('RAP'),
            request.form.get('PPQ'),
            request.form.get('DDP'),
            request.form.get('Shimmer'),
            request.form.get('Shimmer_dB'),
            request.form.get('APQ3'),
            request.form.get('APQ5'),
            request.form.get('APQ'),
            request.form.get('DDA'),
            request.form.get('NHR'),
            request.form.get('HNR'),
            request.form.get('RPDE'),
            request.form.get('DFA'),
            request.form.get('spread1'),
            request.form.get('spread2'),
            request.form.get('D2'),
            request.form.get('PPE')
        ]
        
        user_input = [float(x) if x else 0.0 for x in user_input]
     
        prediction = parkinsons_model.predict([user_input])
        result = "The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease"
    
    return render_template("predict.html", result=result)

# if __name__ == '__main__':
#     flask_app.run(debug=True)

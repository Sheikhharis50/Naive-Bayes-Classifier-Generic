from flask import Flask, render_template, flash, request
from wtforms import (
    Form, TextField, TextAreaField,
    validators, StringField, SubmitField,
    BooleanField,
)
from nbc import nbc
from nbc import message

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    filename = TextAreaField('filename:', validators=[validators.required()])
    inputs = TextAreaField('inputs:', validators=[validators.required()])
    test_per = TextField('test_per:', validators=[validators.required()])
    train_per = TextField('train_per:', validators=[validators.required()])


@app.route("/", methods=['GET','POST'])
def home():
    form = ReusableForm(request.form)
    sms = ''
    acc = 0.0

    if request.method == 'POST':
        print("\nFilename is: ")
        path = request.form['filename']
        print(path)
        print("\nInputs are:")
        inputs = str(request.form['inputs'])
        inputs = inputs.splitlines()
        print(inputs)
        print("\nTesting Percentage is:")
        test_per = str(request.form['test_per'])
        print(test_per)
        print("\nTraining Percentage is:")
        train_per = str(request.form['train_per'])
        print(train_per)
        print("")

        if request.form['filename'] == '' or request.form['inputs'] == '' or request.form['test_per'] == '' or request.form['train_per'] == '':
            return render_template('home.html', form=form, status=False)


        # to use naive bayes classifier
        # needs to create an object of nbc class
        run = nbc()
        # First Read dataset
        run.readDataset(path)
        print("")

        # check if data is properly loaded
        # run.printDataset()

        # shuffle dataset before applying prediction
        run.shuffleDataset()

        # now lets split dataset into training and testing
        # the parameters passing to this methods are:
        # training_percentage and testing_percentage
        run.splitDataset(float(train_per), float(test_per))
        
        predictions = run.getPredictions()
        acc = run.getAccuracy(predictions)

        sms = run.predictClass(inputs)
        # it will give you the predicted class
        # using given inputs

    return render_template('home.html', form=form, status=True, message=sms, accuracy=acc)


if __name__ == "__main__":
    app.run(debug=True)

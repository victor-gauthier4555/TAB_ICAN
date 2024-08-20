from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from backend import  preprocess_dataframe, is_continuous, median_for_condition, final_df

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or request.form.get('column_name') is None:
        return redirect(request.url)

    file = request.files['file']
    column_name = request.form['column_name']

    if file.filename == '':
        return redirect(request.url)

    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath, delimiter = ";")
        df2 = pd.read_csv(filepath, delimiter=";")
        result_df = final_df(df, df2, column_name)

        return render_template('table.html', tables=[result_df.to_html(classes='data', header="true", index=False)])

    return redirect(request.url)



if __name__ == '__main__':
    app.run(debug=True)



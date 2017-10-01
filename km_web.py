from flask import Flask, request, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, FloatField, SelectField
from wtforms.validators import Optional
from flask_wtf.file import FileRequired, FileField
from flask_session import Session

import pandas as pd
import jinja2
import sklearn.preprocessing as pre
import sklearn.cluster as cl

import os

from km import KeplerMapper

## constants
scaling = {'MinMaxScaler': pre.MinMaxScaler()}

clustering = {'DBSCAN': cl.DBSCAN(eps=.5, min_samples=3)}


## Flask config
app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'secret'

# session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# file upload
ALLOWED_EXTENSIONS = set(['npy', 'csv'])
app.config['UPLOAD_FOLDER'] = os.getcwd()


## template
template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TDA</title>
</head>
<body>
    <div class='all_forms'>
        <div class='upload_form'>
            <h3>Upload data</h3>
            <form name='upload_form' action="" method='post' enctype='multipart/form-data'>
                {{ upload_form.hidden_tag() }}
                {{ upload_form.file_path }}
                {{ upload_form.submit }}
            </form>
        </div>
        
        <div class='params_form'>
            {% if current_filename %}
                <h3>Parameters for {{ current_filename }}</h3>
            {% else %}
                <h3>Parameters</h3>
            {% endif %}
            <form name='params_form' action="" method='post'>
                {{ params_form.hidden_tag() }}
                {{ params_form.distance_matrix.label }}
                </br>
                {{ params_form.distance_matrix }}
                </br>
                {{ params_form.filter_value.label }}
                </br>
                {{ params_form.filter_value }}
                </br>
                {{ params_form.scaler.label }}
                </br>
                {{ params_form.scaler }}
                </br>
                {{ params_form.cluster.label }}
                </br>
                {{ params_form.cluster }}
                </br>
                {{ params_form.submit }}
            </form>
        </div>
    </div>
    <h3>Visualization</h3>
    {% if path_html %}
        <div top-image>
            <iframe src="{{ path_html }}" height=800 
                width=1200></iframe>
        </div>
    {% endif %}
</body>
</html>
'''

## forms
class UploadDataForm(FlaskForm):
    file_path = FileField(validators=[FileRequired()])
    submit = SubmitField(default=False)

    def validate(self):
        return True


class SetParamsForm(FlaskForm):
    distance_matrix = StringField('Distance function', default='euclidean')
    filter_value = StringField('Filter', default='sum', validators=[Optional()])
    scaler = SelectField('Scaler', choices=[('MinMaxScaler', 'MinMaxScaler')])
    cluster = SelectField('Cluster algorithm', choices=[('DBSCAN', 'DBSCAN')])

    submit = SubmitField(default=False)

    def validate(self):
        return True

## utils
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_mapping(data, distance, filter, scaler, cluster, path_html):
    km = KeplerMapper(verbose=0)

    scaler_function = scaling[scaler]
    cluster_function = clustering[cluster]

    proj_X = km.fit_transform(data, projection=filter, scaler=scaler_function)
    graph = km.map(proj_X, data, clusterer=cluster_function)
    km.visualize(graph, path_html=path_html)
    return path_html


## views
@app.route('/', methods=['GET', 'POST'])
def index():
    upload_form = UploadDataForm()
    params_form = SetParamsForm()
    t = jinja2.Template(template)
    current_filename = session.get('current_filename', None)
    if request.method == 'POST' and (upload_form.validate_on_submit() or
                                     params_form.validate_on_submit()):

        if upload_form.file_path.data:
            file = upload_form.file_path.data
            df = pd.read_csv(file).values
            session['current_data'] = df
            session['current_filename'] = file.filename
            return t.render(upload_form=upload_form, params_form=params_form,
                            current_filename=current_filename)

        if params_form.submit.data:
            if current_filename is None:
                flash('Upload data first')
            else:
                distance = params_form.distance_matrix.data
                filter = params_form.filter_value.data
                scaler = params_form.scaler.data
                cluster = params_form.cluster.data
                path_html = os.path.join('static', 'tmp.html')
                create_mapping(session.get('current_data'), distance, filter,
                               scaler, cluster, path_html)

            return t.render(upload_form=upload_form, params_form=params_form,
                            current_filename=current_filename, path_html=path_html)


    return t.render(upload_form=upload_form, params_form=params_form,
                    current_file=current_filename)


if __name__ == '__main__':
    app.debug = True
    app.run(port=8082)



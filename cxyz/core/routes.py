from .. import app
from flask import render_template
from .blueprintsserializer import BlueprintsSerializer


@app.route("/")
def index():
    blueprints = BlueprintsSerializer(app).serialize()
    return render_template("index.html", blueprints=blueprints)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

# @app.route("/cv")
# def cv():
#     return render_template("cv.html")

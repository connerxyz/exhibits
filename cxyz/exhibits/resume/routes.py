import os

from cxyz import app
from flask import Blueprint, render_template, url_for
import json
import logging

log = logging.getLogger()

"""
Initialize
"""

TEMPLATE_FOLDER = "./templates"
STATIC_FOLDER = "./static"
PREFIX = "resume"

resume_blueprint = Blueprint(
    PREFIX,
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER
)

"""
Endpoints
"""


@resume_blueprint.route('/')
def resume_exhibit():
    # path = url_for('resume.static', filename='data/skills.json')
    path = "/cxyz/exhibits/resume/static/data/skills.json"
    with open(path, 'r') as f:
        skills = json.load(f)
    # skills.sort(key=lambda x: x['weight'], reverse=True)
    return render_template('resume/exhibit.html', skills=skills)


"""
Register
"""

app.register_blueprint(resume_blueprint, url_prefix="/" + PREFIX)

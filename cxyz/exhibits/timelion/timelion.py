import os

from flask import Blueprint, render_template, url_for

timelion = Blueprint('timelion',
                     __name__,
                     template_folder='./',
                     static_folder='./',
                     static_url_path='/',
                     )
timelion.published = True
timelion.description = "Original artwork for a metal jazz trio."


@timelion.route('/')
def _timelion():
    return render_template('timelion.html')

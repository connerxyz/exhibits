import os

from flask import Blueprint, render_template, url_for

timelion = Blueprint('timelion',
                     __name__,
                     template_folder='./',
                     static_folder='./',
                     static_url_path='/',
                     )


@timelion.route('/')
def _timelion():
    return render_template('timelion.html')

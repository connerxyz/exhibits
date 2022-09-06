import os

from flask import Blueprint, render_template, url_for

minnehack = Blueprint('minnehack',
                      __name__,
                      template_folder='./',
                      static_folder='./',
                      static_url_path='/')
minnehack.display_name = "MinneHack"
minnehack.published = True
minnehack.description = "24 hour hackathon. Great prizes. Totally free."


@minnehack.route('/')
def _minnehack():
    return render_template('minnehack.html')

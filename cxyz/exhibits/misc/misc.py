from flask import Blueprint, render_template
import os

misc = Blueprint('misc',
                 __name__,
                 template_folder='./',
                 static_folder='./',
                 static_url_path='/')
misc.display_name = "Miscellaneous"
misc.published = True
misc.description = "Miscellaneous creations over many years."


@misc.route('/')
def _misc():
    images = os.listdir(os.getcwd() + "/cxyz/exhibits/misc/img")
    images = ["img/" + i for i in images]
    return render_template('misc.html', images=images)

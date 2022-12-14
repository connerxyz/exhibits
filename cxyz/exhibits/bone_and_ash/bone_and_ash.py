from flask import Blueprint, render_template
import os

bone_and_ash = Blueprint('bone_and_ash',
                         __name__,
                         template_folder='./',
                         static_folder='./',
                         static_url_path='/')

bone_and_ash.display_name = "Bone & Ash"
bone_and_ash.published = True
bone_and_ash.description = "A little black moleskin."


@bone_and_ash.route('/')
def _bone_and_ash():
    images = os.listdir(os.getcwd() + "/cxyz/exhibits/bone_and_ash/img")
    images = ["img/" + i for i in images if ".DS_Store" not in i]
    return render_template('bone-and-ash.html', images=images)

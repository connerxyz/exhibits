from flask import Blueprint, render_template
import os

portraits = Blueprint('portraits',
                      __name__,
                      template_folder='./',
                      static_folder='./',
                      static_url_path='/')
portraits.display_name = "Self-portraits"
portraits.published = True
portraits.description = "A collection of self-portraits created using the same paper, pencils, charcoal."


@portraits.route('/')
def _portraits():
    images = os.listdir(os.getcwd() + "/cxyz/exhibits/portraits/img")
    images = ["img/" + i for i in images]
    images.sort()
    return render_template('portraits.html', images=images)

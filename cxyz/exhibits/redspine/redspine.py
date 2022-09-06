from flask import Blueprint, render_template

redspine = Blueprint('redspine',
                     __name__,
                     template_folder='./',
                     static_folder='./',
                     static_url_path='/')
redspine.display_name = "Redspine"
redspine.published = False
redspine.description = "A red-spine notebook. Art that folds in on itself across pages by bleeding through."


@redspine.route('/')
def _redspine():
    return render_template('redspine.html')

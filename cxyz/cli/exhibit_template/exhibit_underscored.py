from flask import Blueprint, render_template

exhibit_underscored = Blueprint('exhibit_underscored',
                                __name__,
                                template_folder='./',
                                static_folder='./',
                                static_url_path='/')

exhibit_underscored.display_name = "exhibit_display_name"
exhibit_underscored.published = False


@exhibit_underscored.route('/')
def _exhibit_underscored():
    return render_template('exhibit.html')

from flask import Blueprint, render_template, url_for

congress = Blueprint('congress',
                     __name__,
                     template_folder='./',
                     static_folder='./',
                     static_url_path='/')
congress.display_name = 'Congress'
congress.published = True
congress.description = "A little app built in 24 hours."


@congress.route('/')
def _congress():
    return render_template('congress.html')

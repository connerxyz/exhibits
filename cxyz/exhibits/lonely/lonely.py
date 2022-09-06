from flask import Blueprint, render_template, url_for

lonely = Blueprint('lonely',
                   __name__,
                   template_folder='./',
                   static_folder='./',
                   static_url_path='/')
lonely.display_name = 'Lonely'
lonely.published = True
lonely.description = "An interactive visualization of original music."


@lonely.route('/')
def _lonely():
    return render_template('lonely.html')

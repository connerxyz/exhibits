from flask import Blueprint, render_template

lines = Blueprint('lines',
                  __name__,
                  template_folder='./',
                  static_folder='./',
                  static_url_path='/')

lines.display_name = "Lines"
lines.published = False
lines.description = 'A "constitutional crisis" visualized using generative art.'


@lines.route('/')
def _lines():
    return render_template('lines.html')

from flask import Blueprint, render_template

speed_and_stability = Blueprint('speed_and_stability',
                                __name__,
                                template_folder='./',
                                static_folder='./',
                                static_url_path='/')

speed_and_stability.display_name = "speed and stability"
speed_and_stability.published = False


@speed_and_stability.route('/')
def _speed_and_stability():
    return render_template('speed-and-stability.html')

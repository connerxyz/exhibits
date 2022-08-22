from flask import Blueprint, render_template

quick_and_dirty_jupyter_notebook_pipelines = Blueprint('quick_and_dirty_jupyter_notebook_pipelines',
                                                       __name__,
                                                       template_folder='./',
                                                       static_folder='./',
                                                       static_url_path='/')

quick_and_dirty_jupyter_notebook_pipelines.display_name = "quick and dirty jupyter notebook pipelines"
quick_and_dirty_jupyter_notebook_pipelines.published = False


@quick_and_dirty_jupyter_notebook_pipelines.route('/')
def _quick_and_dirty_jupyter_notebook_pipelines():
    return render_template('quick-and-dirty-jupyter-notebook-pipelines.html')

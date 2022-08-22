# Initialize flask app
from flask import Flask

app = Flask(
    __name__,
    template_folder="./core/templates",
    static_folder="./core/static"
)

# The app
from flaskext.markdown import Markdown

Markdown(app)  # Support markdown filter in templates

# App routes
from .core import routes
from .exhibits import exhibits

# TODO move this
# Register core static-dependency with template contexts
from .core import StaticDependency


@app.context_processor
def _static_dependencies():
    return {
        'html5_audio': StaticDependency(
            context='static',
            css="html5-audio/html5-audio.css",
            js="html5-audio/html5-audio.js"
        ),
        'jquery': StaticDependency(
            context='static',
            js='jquery/jquery-3.3.1.min.js'
        ),
        'prism_highlighting': StaticDependency(
            context='static',
            css='prism/prism.css',
            js='prism/prism.js'
        ),
        'advanced_zoom': StaticDependency(
            context='static',
            css='advanced-zoom/advanced-zoom.css',
            js='advanced-zoom/advanced-zoom.js'
        ),
        'zoom': StaticDependency(
            context='static',
            css='zoom/css/zoom.css',
            js='zoom/js/zoom.js'
        ),
        'masonry_grid': StaticDependency(
            context='static',
            css='masonry/masonry.css',
            js='masonry/masonry.pkgd.min.js',
            js_init="""
            <!-- initialize masonry -->
            <script>
                window.onload = function(e) {
                    var container = document.querySelector('.masonry-grid');
                    if (container) {
                        var masonry = new Masonry(container, {
                            itemSelector: '.masonry-item',
                            percentPosition: true,
                            gutter: 10,
                        });
                    }
                }
            </script>
            """
        ),
    }


# Register core.utils with all template contexts
from .core import utils


@app.context_processor
def _utils():
    return utils.registry

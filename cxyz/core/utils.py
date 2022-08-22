from flask import url_for

# For utilities that should be available to template contexts
registry = {}


def util(func):
    """Decorator for registering utilities"""
    registry[func.__name__] = func
    return func


@util
def image_gallery(context, images):
    """Assemble a masonry grid image gallery, where each image is also "zoomable"
     from iterable of paths to images"""
    result = '<div class="masonry-grid">'
    for i in images:
        result += (
            '<img class="masonry-item" '
            'src="{}" '
            # 'alt="" '
            'data-action="zoom"'
            '/>'
        ).format(url_for(context, filename=i))
    result += '</div>'
    # TODO: make this an object that encapsulates the HTML/CSS/JS and rendering
    # functionality, refactor static_dependency?
    return result


@util
def audio_gallery(context, files):
    """Assemble an HTML5 audio player to exhibit the audio for each of the given
    file"""
    # TODO: make this an object that encapsulates the HTML/CSS/JS and rendering
    # functionality


@util
def zoomable(context, image):
    return (
        '<img '
        'style="width:100%;" '
        'src="{}" '
        # 'alt="" '
        'data-action="zoom"'
        '/>'
    ).format(url_for(context, filename=image))


"""
{% for audio in site.static_files %}
{% if audio.path contains '/assets/audio/single' %}
<audio controls>
<source src="{{ site.baseurl }}{{ audio.path }}" type="audio/mpeg">
Your browser does not support the audio element.
</audio>
{% endif %}
{% endfor %}

<div class="masonry-grid">
{% for image in site.static_files %}
{% if image.path contains 'assets/img/multiple' %}
<img class="masonry-item" src="{{ site.baseurl }}{{ image.path }}" alt="" data-action="zoom"/>
{% endif %}
{% endfor %}
</div>
"""

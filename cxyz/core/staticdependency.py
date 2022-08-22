from flask import url_for


class StaticDependency(object):

    def __init__(self, context, css=None, js=None, js_init=None):
        self.context = context
        self._css = css
        self._js = js
        self.js_init = js_init

    @property
    def css(self):
        return '<link href="{}" rel="stylesheet">'.format(
            url_for(self.context, filename=self._css)
        )

    @property
    def js(self):
        result = '<script src="{}"></script>'.format(
            url_for(self.context, filename=self._js)
        )
        if self.js_init is not None:
            result += self.js_init
        return result

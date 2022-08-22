DEFAULT_THUMBNAIL = "/static/thumbnail.png"


class Exhibit:

    def __init__(self, blueprint):
        self.name = Exhibit._name(blueprint)
        self.display_name = Exhibit._display_name(blueprint)
        self.thumbnail = Exhibit._thumbnail(blueprint)
        self.url = Exhibit._url(blueprint)

    @staticmethod
    def _name(blueprint):
        return blueprint.name

    @staticmethod
    def _display_name(blueprint):
        if hasattr(blueprint, 'display_name'):
            return blueprint.display_name
        else:
            return blueprint.name

    @staticmethod
    def _url(blueprint):
        return blueprint.name.replace("_", "-")

    @staticmethod
    def _thumbnail(blueprint):
        return "thumbnail.png"


class BlueprintsSerializer:
    """
    Serialize the exhibit blueprints registered with the Flask app instance into
    an array of dicts that can be provided client-side as JSON.
    """

    def __init__(self, app):
        self.app = app

    def serialize(self):
        return [Exhibit(blueprint).__dict__ for blueprint in self.app.blueprints.values()]

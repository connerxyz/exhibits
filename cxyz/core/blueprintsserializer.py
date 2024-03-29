DEFAULT_THUMBNAIL = "/static/thumbnail.png"

import logging

log = logging.getLogger()


class Exhibit:
    """
    A class for processing exhibits.

    - Each exhibit is Flask Blueprint. I.e., a self-contained site submodule.
    - Each exhibit is generated from an exhibit template using the cli (see cxyz/cli).
    - Each exhibit is composed of Flask routes (.py) and static web-content (.html, .css, .js, .png, etc.)
    """

    def __init__(self, blueprint):
        self.name = Exhibit._name(blueprint)
        self.display_name = Exhibit._display_name(blueprint)
        self.published = Exhibit._published(blueprint)
        self.description = Exhibit._description(blueprint)
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
    def _published(blueprint):
        if hasattr(blueprint, 'published'):
            return blueprint.published
        else:
            return False

    @staticmethod
    def _description(blueprint):
        if hasattr(blueprint, 'description'):
            return blueprint.description
        else:
            return "No description provided."

    @staticmethod
    def _url(blueprint):
        return blueprint.name.replace("_", "-")

    @staticmethod
    def _thumbnail(blueprint):
        return "thumbnail.png"


class BlueprintsSerializer:
    """
    Serialize the exhibit blueprints into an array of dicts.

    All exhibit blueprints registered with the Flask app instance are mapped into an array of dicts, which are
    then written to a .json file that's used to render the home page.
    """

    def __init__(self, app):
        self.app = app

    def serialize(self, serialize_unpublished: bool = False):
        """Serialize the exhibit blueprints into a list of dicts used to render the home page.

        Args:
            serialize_unpublished: Should unpublished exhibits also be serialized?

        Returns:
            list[dict]: All the exhibit blueprints as dicts.
        """
        log.info("Serializing exhibits from blueprints...")
        log.info(
            "Including draft exhibits with build." if serialize_unpublished else "Excluding draft exhibits from build."
        )
        result = []
        for blueprint in self.app.blueprints.values():
            if not blueprint.published and serialize_unpublished:
                log.info(f"Building draft exhibit – {blueprint.name}")
                result.append(Exhibit(blueprint).__dict__)
            elif not blueprint.published and not serialize_unpublished:
                log.info(f"Skipping draft exhibit – {blueprint.name}")
            else:
                log.info(f"Building published exhibit – {blueprint.name}")
                result.append(Exhibit(blueprint).__dict__)
        return result

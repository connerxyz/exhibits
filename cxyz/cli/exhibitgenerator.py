import glob
import re
from distutils.dir_util import copy_tree
import os


class ExhibitGenerator:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    EXHIBITS_DIR = BASE_DIR + "/../exhibits/"
    EXHIBIT_TEMPLATE = BASE_DIR + "/exhibit_template"
    EXHIBIT_ROUTES = BASE_DIR + "/../exhibits/exhibits.py"

    def __init__(self, name, display_name=None):
        # Validate exhibit name
        if " " in name:
            raise ValueError("Please provide exhibit name spaces as hyphens.")
        if "exhibit" in name:
            message = "The current exhibit implementation will not work when 'exhibit' appears in the name."
            message += " Please provide a name without 'exhibit' in it, or update the implementation."
            raise ValueError(message)
        self.name = name
        # Underscored version of the name for Python context
        self._name = name.replace("-", "_")
        # Spaces for display-name
        self.display_name = display_name if display_name is not None else name.replace("-", " ")
        self.exhibit_dir = ExhibitGenerator.EXHIBITS_DIR + self._name

    def blueprint_registration(self):
        result = "\nfrom .{} import {}\n".format(self._name, self._name)
        result += 'app.register_blueprint({}, url_prefix="/{}")\n'.format(
            self._name, self.name)
        return result

    def generate(self):
        # Copy exhibit skeleton
        copy_tree(ExhibitGenerator.EXHIBIT_TEMPLATE,
                  self.exhibit_dir)
        # Rename filenames that involve the exhibit name, find and replace within them
        files_to_modify = glob.glob(self.exhibit_dir + "/exhibit*")
        for template_path in files_to_modify:
            # Find and replace within files
            self.find_and_replace(template_path)
            # Rename files
            self.rename(template_path)
        # Now generically named files...
        self.find_and_replace(self.exhibit_dir + "/__init__.py")
        self.find_and_replace(self.exhibit_dir + "/README.md")
        # Now register the blueprint by appending to exhibit's routes
        with open(ExhibitGenerator.EXHIBIT_ROUTES, "a") as f:
            f.write(self.blueprint_registration())

    def find_and_replace(self, template_path):
        with open(template_path, "r+") as f:
            contents = f.read()
            # Note: The order of replacement here matters
            contents = re.sub(r'exhibit_underscored', self._name, contents, flags=re.M)
            contents = re.sub(r'exhibit_display_name', self.display_name, contents)
            contents = re.sub(r'exhibit', self.name, contents)
            f.seek(0)  # Make sure we overwrite, and not append
            f.write(contents)
            f.truncate()

    def rename(self, template_path):
        # Identify the end of the path, e.g. "/exhibit.py"
        desired_path = template_path[template_path.rindex("/"):]
        # Modify the path to replace with exhibit name.
        # E.g. "/exhibit.py" -> "/<exhibit-name.py>".
        if template_path.endswith(".py"):
            desired_path = template_path.replace(desired_path,
                                                 desired_path.replace("exhibit_underscored",
                                                                      self._name))
        else:
            desired_path = template_path.replace(desired_path,
                                                 desired_path.replace("exhibit",
                                                                      self.name))
        os.rename(template_path, desired_path)

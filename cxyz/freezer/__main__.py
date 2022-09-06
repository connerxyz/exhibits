from flask_frozen import Freezer
from cxyz import app
import logging

log = logging.getLogger()


freezer = Freezer(app)

BUILD_DIR = "docs/"
BASE_DOMAIN = "exhibits.conner.xyz"
app.config['FREEZER_BASE_URL'] = f"https://{BASE_DOMAIN}"
app.config['FREEZER_DESTINATION'] = f"../{BUILD_DIR}"
app.config['FREEZER_IGNORE_MIMETYPE_WARNINGS'] = True
app.config['FREEZER_STATIC_IGNORE'] = [
    '**/_/*', # TODO: doesn't work
    '**/_img/*', # TODO: doesn't work
    '**/junk/*', # TODO: doesn't work
    '*.key',
    '*.py',
    'README.md'
]  # TODO figure out a consistent approach / system for all these source media in each exhibit

if __name__ == '__main__':
    log.info("Compiling site to static build using Flask Freezer.")
    log.info(f"Using base URL {app.config['FREEZER_BASE_URL']}.")
    freezer.freeze()
    log.info(f"Adding CNAME for GitHub Pages: {BASE_DOMAIN}")
    with open(f"{BUILD_DIR}/CNAME", "w") as f:
        f.write(BASE_DOMAIN)
    log.info(f"New site build written to {app.config['FREEZER_DESTINATION']}")
from flask_frozen import Freezer
from cxyz import app

freezer = Freezer(app)

app.config['FREEZER_DESTINATION'] = "../docs/"
app.config['FREEZER_BASE_URL'] = "https://connerxyz.github.io/exhibits/"
app.config['FREEZER_IGNORE_MIMETYPE_WARNINGS'] = True
app.config['FREEZER_STATIC_IGNORE'] = [
    '**/_',
    '**/_img',
    '**/junk',
    '*.key',
    '*.py',
]  # TODO figure out a consistent approach / system for all these source media in each exhibit

if __name__ == '__main__':
    freezer.freeze()

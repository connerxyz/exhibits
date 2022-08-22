"""
CXYZ CLI

Usage:
  cxyz generate <exhibit-name>...
"""

from docopt import docopt
from .cli import ExhibitGenerator


def main():
    args = docopt(__doc__)
    if args['generate']:
        ExhibitGenerator(args['<exhibit-name>'][0]).generate()


if __name__ == '__main__':
    main()

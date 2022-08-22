## Core

Core exists to coordinate the exhibition of the exhibits and provide a user-experience "backbone" (e.g. common view
template(s), global navigation, etc.).

## Exhibits

The purpose of this site is to serve a variety of – often independently developed – projects. Each project with its own
purpose, endpoint(s), and dependencies (template, style, js, and assets). In the spirit of *encapsulating what varies*,
we view these heterogeneous projects as homogeneous instances of an *exhibit*.

Each exhibit then becomes a tightly organized unit, decoupled from one another, exposing a consistent interface
to `core`:

- Each exhibit is an import package: include or exclude/exclude exhibit with/without an `import` statement
- Exhibits are decoupled from one another: each exhibit's dependencies are isolated in the source tree, contained within
  the exhibit's import package
- Exhibits expose a consistent interface to `core`: endpoints via `routes.py`

## Adding an exhibit

0. Install the cxyz cli: pip install .
1. cxyz generate <exhibit-name> # Generate and configure a new exhibit

Example:

```
cxyz generate speed-and-stability
```

Consider adding code-highlighting, zoom images, or image gallery CSS and JS imports. See other exhibits for examples on
how to do this.

## issues:

Imagine any page that is not an exhibit...

*How would home/category/topic page know about all the exhibit routes or their categories or topics?*

Imagine trying to "render all images from all exhibits"...

*How to do this?*

Imagine trying to "view all posts containing audio"

*How to do this?*

Imagine building search functionality...

*How to do this?*

Could each exhibit be a class? Providing API for:

- Routes
- Assets: images, audio, video...
- Attributes: categories, topics

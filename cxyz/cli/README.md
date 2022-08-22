# CLI

## Makefile

Provides a CLI for administrating the site.

There is some redundancy with the Makefile, which should be resolved at some point.

The reason both exist at this point, is that the CLI is intended to handle more
sophisticated operations, like generating exhibit skeletons, while the Makefile
exist to capture simple shell commands.

## Exhibit generator

The exhibit generator makes it easy to implement new exhibits.

Each exhibit is a Flask blueprint. The exhibit generator exists to create all
the boilerplate files and configuration necessary for new exhibit Blueprint.

### `exhibit-template`

The exhibit generator uses a "template" exhibit to generate new exhibits.

This template exhibit is `exhibit-template` and exists only to facilitate the
creation of new exhibits – and never to be published itself.

### Quickstart

CLI usage is just below.

Note: Any spaces in the exhibit name should be provided has hyphens.

```
cxyz generator <exhibit-name>
```

### Exhibit URI

The URI for an exhibit is the same the exhibit name provided at generation.

```
cxyz generate exhibit-name
...
open <host>/exhibit-name # Exhibit URL
```

### Display name

The default display name is the same as the exhibit name, with hyphens replaced with spaces.

```
cxyz generate exhibit-name
...
# Display name, the title of the exhibit, will appear on home page as "exhibit name"
```

There will be times you want the display name, e.g. the title of the exhibit,
to be different than the exhibit name.

The exhibit name will be used to name the blueprint's files and URI. While the
display name will be used as the title presented to users in the home page's gallery.

Typically you will want the exhibit name to simpler than the display name.

#### Providing display-name at exhibit generation

You can provide a display-name when generating the exhibit like so – notice this
is also an example of where the exhibit name and URI ('dirent') is much simpler
than the display-name ('*nix directories using c').

```
# An example of an exhibit display name that would make a poor URI
cxyz generate dirent --display-name "*nix directories using c"
```

#### Updating display-name via blueprint

You can also update an exhibit's display-name by editing the `display_name`
attribute in the exhibit's blueprint definition.

```
# exhibit_name.py
...
exhibit_name.display_name = "exhibit name" # Change this
...
```

### Publishing

New exhibits will not appear in the home page's gallery, outside of the
development environment, until they have been published (TODO implement the
production environment with such a feature).

Exhibits are in draft state by default. Publishing is as simple as changing the
`published` attribute in the exhibit's blueprint definition.

```
# exhibit_name.py
...
exhibit_name.published = False # Set to True to publish the exhibit
...
```

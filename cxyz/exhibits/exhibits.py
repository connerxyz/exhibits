from .. import app

from .loita import loita

app.register_blueprint(loita, url_prefix='/loita')

from .lonely import lonely

app.register_blueprint(lonely, url_prefix="/lonely")

from .minnehack import minnehack

app.register_blueprint(minnehack, url_prefix="/minnehack")

from .congress import congress

app.register_blueprint(congress, url_prefix="/congress")

from .portraits import portraits

app.register_blueprint(portraits, url_prefix="/portraits")

from .timelion import timelion

app.register_blueprint(timelion, url_prefix="/timelion")

from .redspine import redspine

app.register_blueprint(redspine, url_prefix="/redspine")

from .func_network import func_network

app.register_blueprint(func_network, url_prefix="/func-network")

from .telescope import telescope

app.register_blueprint(telescope, url_prefix="/telescope")

from .misc import misc

app.register_blueprint(misc, url_prefix="/misc")

from .deca import deca

app.register_blueprint(deca, url_prefix="/deca")

from .codex_ux_and_style_guide import codex_ux_and_style_guide

app.register_blueprint(codex_ux_and_style_guide, url_prefix="/codex-ux-and-style-guide")

from .bone_and_ash import bone_and_ash

app.register_blueprint(bone_and_ash, url_prefix="/bone-and-ash")

from .lines import lines

app.register_blueprint(lines, url_prefix="/lines")

from .error_analysis import error_analysis

app.register_blueprint(error_analysis, url_prefix="/error-analysis")

from .covid_19_gold_data import covid_19_gold_data

app.register_blueprint(covid_19_gold_data, url_prefix="/covid-19-gold-data")

from .helix import helix

app.register_blueprint(helix, url_prefix="/helix")

from .julius import julius

app.register_blueprint(julius, url_prefix="/julius")

from .hello_labs import hello_labs

app.register_blueprint(hello_labs, url_prefix="/hello-labs")

from .happy_place import happy_place

app.register_blueprint(happy_place, url_prefix="/happy-place")

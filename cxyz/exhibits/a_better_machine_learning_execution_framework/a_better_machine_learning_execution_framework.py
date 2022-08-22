from flask import Blueprint, render_template

a_better_machine_learning_execution_framework = Blueprint('a_better_machine_learning_execution_framework',
                                                          __name__,
                                                          template_folder='./',
                                                          static_folder='./',
                                                          static_url_path='/')

a_better_machine_learning_execution_framework.display_name = "a better machine learning execution framework"
a_better_machine_learning_execution_framework.published = False


@a_better_machine_learning_execution_framework.route('/')
def _a_better_machine_learning_execution_framework():
    return render_template('a-better-machine-learning-execution-framework.html')

from flask import Blueprint, render_template

launching_a_flask_app_using_aws_elastic_beanstalk = Blueprint('launching_a_flask_app_using_aws_elastic_beanstalk',
                                                              __name__,
                                                              template_folder='./',
                                                              static_folder='./',
                                                              static_url_path='/')

launching_a_flask_app_using_aws_elastic_beanstalk.display_name = "launching a flask app using aws elastic beanstalk"
launching_a_flask_app_using_aws_elastic_beanstalk.published = False


@launching_a_flask_app_using_aws_elastic_beanstalk.route('/')
def _launching_a_flask_app_using_aws_elastic_beanstalk():
    return render_template('launching-a-flask-app-using-aws-elastic-beanstalk.html')

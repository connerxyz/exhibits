SHELL          = /bin/bash

# Misc dev utilities

test:
	echo ${URL}
	echo ${NAME}
	echo ${COMMIT}

clean-logs:
	rm ./cxyz/log/*

# Serve locally for development purposes
dev: export FLASK_APP=cxyz
dev: export FLASK_DEBUG=True
dev: export FLASK_RUN_PORT=5001
dev:
	poetry run flask run

# Delete any prior build
clean:
	rm -rf docs/

# Build static site using Flask Freezer
build:
	poetry run python cxyz/freezer
	echo "New site build written to docs/"

# Deploy static build to GitHub Pages
deploy: build, clean
	git add docs/
	git commit "New site build."
	git push

# Prepare package for Elastic Beanstalk deployment
bundle-ebs:
	git archive -v -o cxyz.zip --format=zip HEAD

# Deploy via Elastic Beanstalk
deploy-ebs:
	eb init -p python-3.7 conner.xyz --region us-east-1
	eb create cxyz-env

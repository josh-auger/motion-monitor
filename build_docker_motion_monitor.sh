#!/bin/bash

#docker build --rm -t jauger/motion-monitor:latest -f ./Dockerfile .
docker build --rm --build-arg HTTP_PROXY=http://proxy.tch.harvard.edu:3128 -t jauger/motion-monitor:latest -f ./Dockerfile .
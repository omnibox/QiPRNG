#!/bin/bash

git remote add QiPRNG https://github.com/Aaron-Gregory/QiPRNG.git
git fetch QiPRNG
git checkout main
git merge QiPRNG/main

#!/usr/bin/bash

sed -i '' '1s/^/#/' .gitignore

bundle exec jekyll build 

git stash push

git checkout gh-pages

git stash pop{1}

sed -i '' '1s/^.//' .gitignore

cp -r _site/* .
rm -rf _site

git add .
git commit -m 'update'

git push origin gh-pages

git checkout master

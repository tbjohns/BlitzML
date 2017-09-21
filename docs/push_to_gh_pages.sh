make clean
make html
cd _build
mv html html.new
git clone -b gh-pages https://github.com/tbjohns/BlitzML.git html
cd html
git rm -r .
mv ../html.new/{.,}* .
touch .nojekyll
git add .
git commit -m "updated docs"
cd ..
rm -r html.new
echo "To complete push, cd to _build/html and run 'git push origin gh-pages'."


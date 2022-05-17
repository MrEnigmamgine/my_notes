# How to initialize a git repository via cli
First make sure the directory exists, then open it with CLI
```
git init
git add -A
git commit -m "Repository initialized"
git remote add origin git@github.com:MrEnigmamgine/<project_nmae>.git
git push -u -f origin master
```
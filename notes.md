# How to initialize a git repository via cli and push to Github

- Make sure the repository exists on github
- Make sure the root folder matches the repository name
- open the root folder in CLI

```
git init
git add -A
git commit -m "Repository initialized"
git remote add origin git@github.com:MrEnigmamgine/<repository_name>.git
git push -u -f origin master
```
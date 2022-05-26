# Git

### How to initialize a git repository via cli and push to Github

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

### quickpush function

Most shells include the ability to make persistant aliases.  For bash it's a `.bash_aliases` file in your home directory. `~/.bash_aliases`.
We can hijack that functionality to also make shell functions, which are really just fancy multi-line aliases that accept arguments.
Add the following into your aliases file:
```
function quickpush () {
    git add -A
    git commit -m "$*"
    git push
}
```
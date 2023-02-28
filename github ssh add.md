## Configuring Git
When you first install git, and before you can start using it with github there are some configuration tasks that need to be completed beforehand.

[Git - First-Time Git Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
> You can view all of your settings and where they are coming from using:
> 
>     $ git config --list --show-origin


> ### Your Identity
> 
> The first thing you should do when you install Git is to set your user name and email address. This is important because every Git commit uses this information, and it’s immutably baked into the commits you start creating:
> 
>     $ git config --global user.name "John Doe"
>     $ git config --global user.email johndoe@example.com

## Generating an ssh key pair
>In terminal:
```
ssh-keygen -t ed25519 -C "{your_email@example.com}" -f ~/.ssh/{key_name}
```
>
    -t dsa | ecdsa | ecdsa-sk | ed25519 | ed25519-sk | rsa
            Specifies the type of key to create.  The possible values are
            “dsa”, “ecdsa”, “ecdsa-sk”, “ed25519”, “ed25519-sk”, or “rsa”.
     -C comment
             Provides a new comment.
     -f filename
             Specifies the filename of the key file.

This will generate two files in `~/.ssh`, one is the private key and the other is the public key (denoted by the .pub extension).  

You will need to add the private key to your ssh environment.
> 
    ssh-add ~/.ssh/{key_name}


> Add the SSH key to your account on GitHub. For more information, see "[Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)."
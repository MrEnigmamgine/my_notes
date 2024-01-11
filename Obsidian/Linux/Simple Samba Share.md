
Samba is configured from a single file that lives in `/etc/samba/smb.conf`

Add configuration information to the end of the file. Something like this worked for me.
```
[share_name]
   comment = share_description
   path = /path/to/directory
   browsable = yes
   valid users = crux
   force group = mediacenter
   guest ok = no
   read only = no
   create mask = 0765
```

It's important to note that samba users are not shared with the system, to create a samba user:
```sh
sudo smbpasswd -a your_unix_username
```


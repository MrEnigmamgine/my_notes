# Users
Existing user configuration is stored in `/etc/passwd`

### Add a user
`sudo adduser {username}`
The following options are good for service accounts
`--system `
`--no-create-home `
`--disabled-login `
`--uid {uid}`

### Delete a user
`sudo userdel {username}`

### Change password
`sudo passwd {username}`

## Groups
`/etc/group`
### Add a group
`sudo addgroup {groupname}`
Optionally:
`--gid {gid}`

### Delete a group
`sudo groupdel {groupname}`

# Permissions

Permissions in linux is based on 3 sets of levels.
`{user} {group} {everyone}`

The levels are the sums of the following permissions:

| \# | Permission |
| ---- | ---- |
| 4 | read |
| 2 | write |
| 1 | execute |
Above 4 is the sum of the above.  

| \# | Permission |
| ---- | ---- |
| 5 | read + execute |
| 6 | read + write |
| 7 | read + write + execute |

They can also be expressed symbolically  in `rwx` format:
	`-rwxrw-r--`

To change permissions on a file or folder:
```
sudo chmod {permission} {path}  #-R #Recursive flag
```
where `{permission}` can be expressed numerically (eg. 755) or as symbolic modification to existing permissions (eg. g+w adds Write permission to the Group bit.)

### Add a user to a group.
```
sudo usermod -aG {group} {user}
```
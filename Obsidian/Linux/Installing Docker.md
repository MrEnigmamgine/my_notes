
There were some unofficial packages. In case they are installed we will remove them with
```sh
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```
... most likely none of these are installed.

Docker wants us to use their official ppa.  Do to enable it we use
```sh
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

With docker's official ppa in place we can install docker using
```sh
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

- `docker-ce`: Installs the Docker Community Edition
- `docker-ce-cli`: Installs the Docker command-line interface (CLI)
- `containerd.io`: Installs Containerd
- `docker-buildx-plugin`: Installs the Docker Buildx plugin, which extends the capabilities of the standard `docker build` command to support additional platforms and features.
- `docker-compose-plugin`: Enables support for docker-compose YAML files.

### Verify installation
```sh
sudo docker run hello-world
```
This command downloads a test image and runs it in a container. When the container runs, it prints a confirmation message and exits.

### Automatic startup
If you'd like to have docker service running at startup run
```sh
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```
This command registers docker with the `systemd` init system.

Add NVIDIA repositories
```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install NVIDIA container toolkit
```sh
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure docker and containerd to use the toolkit
```sh
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart docker
sudo systemctl restart containerd
```

### Using NVIDIA in docker

Via docker-compose
```yaml
  service_name:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
              #driver: nvidia #for forcing nvidia driver
```


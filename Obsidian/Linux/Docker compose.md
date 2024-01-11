## Updating images
You can use docker compose to easily update containers.
```sh
docker compose pull
docker compose up -d
```
This will first pull the images and issuing the `up` command will only recreate the ones that were updated.


This might also work
```
docker compose up --detach --remove-orphans --pull=always
```
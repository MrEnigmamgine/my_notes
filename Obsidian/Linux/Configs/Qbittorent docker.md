Docker Compose example
```docker-compose
  ## Torrent client with web interface

  qbittorrent:
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    environment:
      - PUID=13001
      - PGID=13000
      - UMASK=002
      - TZ=${TIMEZONE}
      - WEBUI_PORT=8080
    volumes:
      - ${CONF_DIR}/config/qbittorrent-config:/config
      - ${DATA_DIR}/data/torrents:/data/torrents
    ports:
      - "8080:8080"
      - "6881:6881"
      - "6881:6881/udp"
    restart: unless-stopped
```

For blocking copywrite crawlers copy `.p2p` file from https://github.com/waelisa/Best-blocklist/blob/main/wael.list.p2p

Place file in `/config` and define the file under `Options > Connections > IP Filtering`
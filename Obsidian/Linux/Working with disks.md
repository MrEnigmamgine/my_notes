
## Identifying devices

In linux, everything is treated as a file. This includes devices such as disks. By convention these "files" are stored in `/dev/`.  However, it can be difficult to sort out which device is which, or how it's being used by the system.  The `lsblk` or `(list block devices)` utility can be helpful here.

```bash
lsblk
```

The utility should output something like this:
```
NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
sda      8:0    0   1.4T  0 disk /mnt/d
sdb      8:16   0   1.8T  0 disk
sdc      8:32   0 223.6G  0 disk
├─sdc1   8:33   0   512M  0 part /boot/efi
└─sdc2   8:34   0 223.1G  0 part /
```
This output lists the block devices and the partitions on the system, as well as their mount points.  In this example, `sdc` is my system disk and `sdc2` is my system partition.

---
Additionally, disks can be listed using the `fdisk` utility, which will require administrator permissions.

```
sudo fdisk -l
```

Which outputs a list of entries like this one:
```
Disk /dev/sdc: 223.57 GiB, 240057409536 bytes, 468862128 sectors
Disk model: INTEL SSDSC2CT24
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: gpt
Disk identifier: BC369587-8005-4D50-A4B2-D48205B5DF36

Device       Start       End   Sectors   Size Type
/dev/sdc1     2048   1050623   1048576   512M EFI System
/dev/sdc2  1050624 468860927 467810304 223.1G Linux filesystem
```

## Creating new partitions

We'll `fdisk` utility from earlier. Fdisk is an interactive program, meaning you will navigate through a series of menus to create a list of commands for fdisk to carry out. It won't carry out those commands until you tell it to, however.

To start, run fdisk against the device you wish to work with.
```
sudo fdisk /{device_path}
```

From here, the world is your oyster, configure the drive how you want.  For my own purposes, however, I will record my workflow.

| Key | Description | Note |
| --- | --- | --- |
| `p` | Print info | |
| `g` | New GPT partition table | Wipes any existing partitions |
| `n` | New partition | interactive, defaults usually fine |
| `w` | Write changes to disk | |

### Formatting partitions

So far we've created a partition, but it's not useful without a filesystem. To resolve this we'll use `mkfs` (make-filesystem) utilities.  ext4 filesystems are good for partitions that will only be used by linux, for everything else there's exfat.

> ⚠  Be sure to use the **partition** and not the device!
```
sudo mkfs.ext4 /{partition_path}
```

> If you run into trouble with making an exfat filesystem, it might be necessary to install the necessary packages.
```
sudo apt install exfat-utils exfat-fuse
```

## Mounts

### Mounting
```
sudo mount {device_path} {mount_point}
```
> Note: this mount method does not persist on reboot

### Automatic Mounting

Linux handles the automatic mounting of filesystems through the `/etc/fstab` file.  Adding an entry in the fstab will cause linux to mount the filesystems line-by-line.
> ⚠ An incorrect fstab can cause the system not to boot properly

Here are the fields needed for a fstab entry in syntax order:

|**Field**|**Description**|**Recommendation**|
|---|---|---|
|device|The device/partition (by /dev location or UUID) | UUID is consistent if things get shuffled around physically |
|mount point| an existing directory on the system | somewhere in `/mnt` |
|fs type| | `auto` |
|options|Mount options (see the man page for mount).| `defaults` |
|dump|Enable or disable backing up of the device/partition (the command _dump_).| `0` |
|pass|Controls the order in which _fsck_ checks the device/partition for errors at boot time. | root device should be 1. Others should be 2, or 0|

#### Finding UUIDs
The easiest way is probably
```
lsblk -f
```
However, this also works
```
ls /dev/disk/by-uuid/ -l
```
### Unmounting
```
sudo umount {device_path}
```
or
```
sudo umount {mount_point}
```

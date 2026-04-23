# Docker

## 1. Clone IsaacLab version 2.1.1
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd Isaaclab
git checkout v2.1.1
```

recommended: remove origin of IsaacLab to avoid accidentally pushing to it
```bash
git remote remove origin
```

check if remote is removed (should return empty)
```bash
git remote -v
```

## 2. Copy Customized Docker Environment
In the /docker folder of this project, the files related to docker can be copied and (re)placed into the /docker folder of isaac lab.

Dockerfile.base: install sru and plr packages, customize setup

.env.base-plr: create custom profiles to be able to distinguish the individual singularity images on cluster

## 3. Download Dependency Repos

Inside the Isaaclab folder, run the following commands to download all the dependencies (isaaclab_nav_task is not a dependency, but included for reference on how to use SRU)

```bash
git clone https://github.com/leggedrobotics/sru-navigation-sim.git source/isaaclab_nav_task
git clone https://github.com/leggedrobotics/sru-navigation-learning.git source/rsl_rl
git clone https://github.com/yhostettler/plr source/plr
```

## 4. Start container
```bash
docker/container.py start --suffix plr
```

## 5. Notes
Best practice/Avoid permission errors: Write new code/Commit code outside the container and run code inside the container as the container user is root
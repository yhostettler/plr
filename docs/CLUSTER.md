# Cluster

## 1. Set up Docker Environment
Go through all the steps in [`DOCKER.md`](DOCKER.md)

## 2. Copy Customized Cluster Environment
In the /docker/cluster folder of this project, copy the files and (re)place them into the /docker/cluster folder of isaac lab.

Important: Adjust the files where needed. Add WandB API Key, ensure that correct email is set for the END notification email. Ensure that euler is set in .ssh/config -> to use ssh without password request when pushing the image to the cluster.

## 3. Build Docker image
```bash
./docker/container.sh start --suffix plr
```

## 4. Push to cluster (converts to singularity automatically)

```bash
./docker/cluster/cluster_interface.sh push base-plr
```
This may take a very long  time (~2hrs, even 3-4 hrs when not deleting unneeded repos before)


## 5. Submit training job
```bash
./docker/cluster/cluster_interface.sh job base-plr \
"--task Isaac-PLR-LOC-Anymal-D-Static-v0" \
"--num_envs 2048" \
"--headless" 
```

## 6. Monitor job (in euler ssh session)
```bash
myjobs
```
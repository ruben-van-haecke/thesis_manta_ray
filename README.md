# thesis_manta_ray

## Installation
Run the following commands to install the required packages:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## To freeze the requirements
```
pip freeze > requirements.txt
```

## Connection to the HPC
```
ssh -i //home/ruben/.ssh/vsc_hpc vsc45099@login.hpc.ugent.be
```

```
module swap cluster/doduo
qsub hpc_job.sh
```
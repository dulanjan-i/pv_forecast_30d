# tl;dr
- Avoid execution of scripts and applications on the login node.
- Execution of scripts and applications is only allowed in containers.
- Containers must be executed via slurm.
- Containers are able to mount `/home/$USER` and `/shared/$USER`.
- Containers must be started with `-c` flag.
- Internet access/firewall rules depend on your institute.
- External dependencies (pip, anaconda, npm, r...) must be checked by Sonatype.
- Containers have no network connection by default. Network connection is explicit.
- There is no backup of your data. In the unlikely event of an HPC failure, data may be lost.
- When text is written between `[` and `]` in this document, it will indicate that this part is optional. The brackets itself are not part of the text either.
- When text is written between `<` and `>` in this document, it will indicate that this part is necessary and needs to be replaced with a valid value. The brackets itself are not part of the text.
- If you are not familiar with Linux, read `Linux Basics` first. Run some examples manually without any other workflow involved to get a feeling what is a shell and how Sonatype, Singularity and Slurm play together. Just you and the shell (and probably this documentation).
- See the `Step by step guide` at the end of this documentation for an example.

# Linux Basics
If you are not familiar with Linux, here is some basic knowledge. This will cover some basics to understand what is happening in this document. 
### General
- You will probably read terms like `shell`, `bash`, `cli`, `cmd`, `command line` or `terminal`. In our context they mean the same thing. Here you can run commands and see output of your programs. You can edit files and operate with the base system.
- The HPC has no graphical user interface (GUI). You need to use the shell to access and use it.
- You can use many command line tools to connect via SSH. Examples are: PuTTY, Powershell, the shell on other Linux Systems, the Terminal in VS Code (which is probably Powershell by default), Git Bash.
- You must connect via `ssh` to the HPC. See "Common commands and concepts" for an example.
- Integration in other software like CI/CD or accessing the file system in VS Code is mostly possible. Either you will find tutorials on the internet or you can ask colleagues and administrators for help.
- In this document you can see `$USER` sometimes. On shell `$` indicates a variable which will be replaced when executing. `$USER` is always available an will be replaced with your username.
- For writing small programs on the fly without the need for any environment you can use shell scripts. You need (small) shell scripts to run jobs on the HPC (see `sbatch`).

### Common commands and concepts
- `pwd`: Print working directory. Your current path your are in and executing commands.
- `ls`: List of files in your current directory. `ls -l [/path]` and `ls -la [/path]` are common parameters.
- `whoami`: Shows your username. Is the same as `echo $USER`.
- `cd`: Change directory. `cd /shared` will set your current directory to `/shared`. `cd ~` will set your current directory to your home directory.
- `cp`: Copy files and directories. Pattern is `cp [/path/]src.txt [/path/]dst.txt`. For directories you need to use the parameter `-r` (recursive), `cp -r [/path/]src-directory [/path/]dst-directory`.
- `mv`: Moves files and directories. Pattern is `mv src dst`. Can also be used to rename.
- `rm`: Will delete files and directories. Pattern is `rm file.txt`. Use `rm -r` for directories.
- `cat`: Will print contents of file. Avoid to print binary data like images.
- `wget`: Will download the content of a URL and save it to a file. `wget ubuntu.com`.
- `git`: The command line version of git. https://git-scm.com/doc
- `nano`: Simple terminal editor. https://en.wikipedia.org/wiki/GNU_nano see "Control keys" for usage.
- `vim`: Advanced terminal editor. https://www.linuxfoundation.org/blog/blog/classic-sysadmin-vim-101-a-beginners-guide-to-vim
- `ssh`: Connect to a remote server with a specific user. `ssh username@172.29.255.253`. Will prompt for password if necessary.
- `scp`: Used to transfer files between current system and remote server. https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/
- `rsync`: Used to sync folders across the network, has more features than `scp`. Also works via `ssh`. https://linuxize.com/post/how-to-use-rsync-for-local-and-remote-data-transfer-and-synchronization/
- `/bin/hostname`: Prints hostname of server.
- home directory: This is the directory you will be directed when logging in. Usually it is something like `/home/username`. Only you can access this directory. No other user can see your files.
- When selecting files and directories (cp, mv, rm, scp...) you can mostly also use pattern matching (regex) for specifing the path. This is an advanced but mighty feature, use with care (especially in combination with `rm`!). https://www.linuxjournal.com/content/pattern-matching-bash
- Almost all Linux commands allow the use of parameters. With parameters you can further specify the settings for running the command. In most cases `<command> help`, `<command> -help` or `<command> --help` will print informations about possible parameters.
- More useful commands you can take a look at: `grep`, `find`, `tail`, `head`, `|` https://www.geeksforgeeks.org/piping-in-unix-or-linux/, `>` and `>>` https://www.geeksforgeeks.org/input-output-redirection-in-linux/

# Connect to HPC
### At DBFZ
- ssh into `dbfz-linux-user01.leipzig.dbfz.de`. Since ldap is enabled, your regular credentials are working.
- From there ssh into `172.29.255.253` with your provided credentials.
### At Thünen
- ssh into `entry-dbfz.kida.thuenen.de` with your provided credentials.


# /home and /shared
- Your home directory `/home/$USER` is not shared between nodes. On every node, you have a separate home directory.
- Your shared directory `/shared/$USER` is shared between nodes.
- If you want to share files between jobs (scripts, containers) put them into `/shared/$USER`.
- If jobs write to `/home/$USER` and you need to access these files, you have to move them explicit to `/shared/$USER` in your scripts.

# Network settings
- Containers have no network access by default.
- Network access is explicit. See general examples below.
- When you need network access, you have to pass options to singularity containers.
- Examples are valid for the HPC of DBFZ.
- You cannot use other networks other than your institute.
- If you need to access ressources in your institute or specific links, reach out to your admins first. KIDA is not responsible for that.
### DBFZ
`singularity exec --net --network dbfz`
### BVL
`singularity exec --net --network bvl --dns 172.26.0.65`
### JKI
`singularity exec --net --network jki --dns 172.27.12.126`
### BfR
`singularity exec --net --network jki --dns 172.27.37.6`
### Thünen
`singularity exec --net --network thuenen --dns 172.21.0.1`
### MRI
`singularity exec --net --network mri --dns 8.8.8.8`

# Sonatype
Sonatype improves the supply-chain-security. In most cases packages from pip, conda, r are downloaded with no further checking.

When working on the HPC all your your third-party packages must be checked by Sonatype.
### pip config
An up-to-date `pip.conf` is placed in `/shared/configs/<your-institute>/pip.conf`. If you need the mirror URL for any reason, you can find it there.
### anaconda config
In `/shared/definitions/<your-institute>/conda_minimal_base.def` is the up-to-date conda mirror configured. If needed, you can check all steps there to exclusively use Sonatype mirrors for anaconda.
### R config
In `/shared/definitions/<your-institute>/r_template.def` is the up-to-date R mirror configured. See `%help` section for more informations.
### Other mirrors
If you require additional mirrors, please contact the Sonatype contact person at your institute. They should be able to provide you with all the necessary information.

# Singularity
This will only cover the basic commands to create and execute a container. If you need further functionality take a look at the officals docs first, if not clear reach out to administrators and describe your problem.
### Official docs
https://docs.sylabs.io/guides/latest/user-guide/

### Build Container

For building a container, you need a definition file. Similar to a Dockerfile, you specify your requirements in the container, such as the base operating system, installed packages, third-party packages, and configurations. For more details, see the [Singularity definition file documentation](https://docs.sylabs.io/guides/latest/user-guide/definition_files.html).

In `/shared/definitions/<your-institute>`, you can find templates with the most basic configurations.

#### **Security Consideration**
On the HPC, we use the `srun` command to build Singularity containers for security reasons. Using `srun` ensures that the build process runs within a Slurm-allocated environment, preventing unauthorized resource usage or privilege escalation that could occur when building containers directly on a shared login node. This approach provides an additional layer of isolation and compliance with HPC policies.

#### **Command to Build a Container**
The basic command to build a container on the HPC is:

```bash
srun --pty singularity build --fakeroot /shared/<user_name>/test.sif /shared/<user_name>/nginx.def
```

### Execute Container
The following examples will start the container on the login node. This is only recommended for small tests of functionality. For running anything else (calculations, starting (temporary) servers, using llm models...) please use slurm to start containers on nodes.

There a basically three subcommands you can use to start the container:
1. `singularity run image.sif` will run the command specified in `%runscript` in the definition file.
2. `singularity exec image.sif command` will run your given `command`.
3. `singularity shell image.sif` will open a new shell inside the container.

### Specific notes for our HPC
- You can probably use one of the following headers in your definition file to build an image:
```
BootStrap: library
From: ubuntu:22.04
```
This will pull an up-to-date image from ubuntu. Of course you can specify the version you will need. Make sure the version still get updates.

```
BootStrap: localimage
From: /path/to/image.sif
```
This will use an already built image as source. This helps to reduce build time when trying out new packages/stuff. So for example you first build an image with your basic needs. Then create a new definition file which will use this image and only install only few additional packages to try them out.
- Regardless of your header choosen, it is recommended to update all packages and clean. It should look something like:
```
apt-get update
apt-get -y upgrade
apt-get install -y your-needed-packages (optional)
apt-get -y autoremove
apt-get -y clean
```
- All containers started are enforced to use the flag `-c`. This will increase security aspects like no auto binding directories or no shared tmp directory between containers.
- Because of the `-c` flag, `--bind /shared/$USER:/shared/$USER,/home/$USER:/home/$USER` is necessary and also enforced. This will make your home directory and the shared directory available in the container.
- When using GPUs you need to specify the `--nv` flag.
- By default network is disabled. When network access is needed you need to specify it like described under `Network settings` in this document.

### Examples
- `singularity exec image.sif whoami`: Will print your username.
- `singularity exec image.sif wget ubuntu.com`: Will fail, since there is no network available.
- `singularity exec --nv image.sif nvidia-smi`: Will show output of `nvidia-smi`, only works on GPU nodes.
- `singularity exec --net --network dbfz image.sif wget ubuntu.com`: Will download index.html from ubuntu.com.

# Recommendations for what belongs where
- A typical software project includes your source code, scripts, assets and depends on external packages. Probably you have input data to process.
- Avoid to copy your source code into images. Images should be as generic as possible for reusability. Your code should be in `/home/$USER` or `/shared/$USER`.
- Images should include the complete environment to run your script without installing any further dependencies on the fly.
- If your input data are files, you should place them in `/home/$USER` or `/shared/$USER`. Strictly avoid to put your data in the image.
- If your input data depends on online ressources, like a public database, you need to start the container with network access. The firewall of your institute may block the access.

# Slurm
This will only cover the basic commands to start jobs on the cluster. If you need further functionality take a look at the offical docs first, if not clear reach out to administrators and describe your problem.

Slurm will schedule your job automatically on the cluster when your requested ressources are available. There is no prioritization of jobs or users.
### Official docs
https://slurm.schedmd.com/documentation.html

### sacct/squeue
Will get informations of your jobs. You can run these commands wihtout any other parameter. See documentation for more options if needed.

### sinfo
Will get informations about nodes. Will basically show if nodes available. See documentation for more options if needed.

To get informations about how many ressources are free on each node, you can use `scontrol show node`.

### srun
Will run a job on a node in foreground. Shell is blocked during execution and you will see output directly. Most important parameters propably are:
- -c, --cpus-per-task=ncpus number of cpus required per task
- --mem=MB minimum amount of real memory
- -t, --time=minutes time limit
- -p, --partition=partition partition requested
- --gres=list required generic resources
- --pty run task zero in pseudo terminal
- For everything else, please see the documentation.

### sbatch
Will submit a batch script to slurm. The script will be executed on nodes, when possible. Status can be checked with `sacct` or `squeue`. Shell is not blocked. Instead of using parameters for sbatch, you can set settings in the batch file directly. See the example below. For all options see the official documentation.

You can simply run the file with `sbatch job.sh`.

```
#!/bin/bash
#SBATCH --output=/shared/<user>/%j.out # %j is the job id
#SBATCH --error=/shared/<user>/%j.err
#SBATCH --mem=30G
#SBATCH --job-name job_name
#SBATCH --partition=gpuh100
#SBATCH --mail-type=ALL # will send informations about the job to your mail address
#SBATCH --mail-user=your-email-address

mv /shared/$USER/dosomething.py ~
singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network none /shared/<user>/image.sif python3 dosomething.py
mv dosomething.log /shared/<user>/
rm ~/dosomething.py
```
#### Job Array
With job arrays, it is possible to execute a process multiple times and specify different parameters. The current index can be used with the `SLURM_ARRAY_TASK_ID` variable.
```
#!/bin/bash
#SBATCH --job-name=job-array
#SBATCH --output=/shared/<user>/%A_%a.out # %A is the job id, %a is the array index
#SBATCH --array=1-10

echo "SLURM_ARRAY_TASK_ID: "$SLURM_ARRAY_TASK_ID
singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network none /shared/<user>/image.sif <command> $SLURM_ARRAY_TASK_ID
```

#### Networks in sbatch
- `singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network none /shared/<user>/image.sif <command>`: Will start with no network.
- `singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> /shared/<user>/image.sif <command>`: Will start with DBFZ network. Works only for DBFZ employees on the DBFZ HPC.
- `singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network bvl --dns 172.26.0.65 /shared/<user>/image.sif <command>`: Will start with BVL network. Works only for BVL employees.
- `singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network jki --dns 172.27.12.126 /shared/<user>/image.sif <command>`: Will start with JKI network. Works only for JKI employees.
- `singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network bfr --dns 172.27.37.6 /shared/<user>/image.sif <command>`: Will start with BfR network. Works only for BfR employees.
- `singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network thuenen --dns 172.21.0.1 /shared/<user>/image.sif <command>`: Will start with Thuenen network. Works only for Thuenen employees.
- `singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> --net --network mri --dns 8.8.8.8 /shared/<user>/image.sif <command>`: Will start with MRI network. Works only for MRI employees.

#### Default values
If you don't define the ressource needs of your job, only one CPU per job is allocated. If your job needs more than one CPU and it is free, your job continues, otherwise is blocked or rejected. Memory of any size is not guaranteed. Jobs which define ressource needs will take away the usable memory of your job.

### Spefic notes for our HPC
- srun and sbatch must use singularity for executing your tasks. Also you can only use the network of your institute. This behaviour is enforced. If you, however, find a way to exploit this, you are requested to contact administrators and not to use this exploit.

### Examples
- `srun singularity exec /shared/<user>/image.sif /bin/hostname`: Will print hostname of node.
- `srun -p gpuh100 singularity exec /shared/<user>/image.sif nvidia-smi`: Will be executed on gpu node, but will fail because `--nv` flag is missing.
- `srun -p gpuh100 singularity exec --nv /shared/<user>/image.sif nvidia-smi`: Will be executed on gpu node, will print informations about gpus.
- `srun -c 10 --mem 10G --time 1:00:00 singularity exec /shared/<user>/image.sif <command>`: Will block 10 cores and 10 GB RAM for your job. Job will be killed if not finished after 1 hour.
- `srun -p gpuh100 --gres=gpu:1 singularity exec --nv /shared/<user>/image.sif <commandY`: Will block one gpu for your job.
- `sbatch job.sh`: Will submit job.sh to slurm.

# Interactive Shell via slurm
You can run an interactive shell on nodes with `srun --pty singularity shell image.sif`. This is just the basic command, of course you can extend it with every option/flag previously shown.

# Prebuild containers
### Matlab (DBFZ only!)
A Matlab image can be provided for DBFZ users. Since we have to keep track of the license, please contact bernd.hassfurther@dbfz.de for access.

### Jupyter Notebook
Your institute have to configure specific firewall rules. The following example is valid for DBFZ at the moment.

For DBFZ users it is possible to use some specific ports for Jupyter Notebook.
- `/shared/definitions/dbfz/notebook_base.def` is the base configuration for jupyter notebook. You probably don't want to edit this, but you can take a look.
- `/shared/definitions/dbfz/notebook_base.sif` is the image of the definition file.
- `/shared/definitions/dbfz/notebook_template.def` is meant to be copied and install extra dependencies you need.
- You can use the ports 8800-8810.
- Start your container with `srun --pty singularity shell --net --network dbfz image.sif`.
- By default the notebooks started are public, use a password if needed.
- You can create the config for using password with `jupyter notebook password`.
- Start the notebook with `jupyter notebook --port 88xx -–ip 0.0.0.0`. Replace `xx` with your desired port.
- You should see a line like `http://dbfz-hpc23-cnode1:8800/tree`.
- Extend the hostname with `.hpc.leipzig.dbfz.de`. This is the link which you can use in the browser to access the notebook. If you have set a password, it will be prompted.

# Step by step guide
1. **How to get an account?**
- If you want access to the HPC, please use these contacts.
  - BVL: boris.orywahl-wild@bvl.bund.de
  - DBFZ: bernd.hassfurther@dbfz.de
  - JKI: emanuel.hesse@julius-kuehn.de
  - Thünen: marcus.oppermann@thuenen.de

2. **How to connect?**
- **DBFZ**
  - On your computer, open PuTTY.
  - Host Name: `dbfz-linux-user01`, Port: `22`, Connection type: `SSH`, Click the button `Open`.
  - Login with your Windows-Credentials.
  - Run `ssh <user>@172.29.255.253`. The account name is provided in step 1. If this is the first time you login, you will asked to change the initial password.
- **Thünen**
  - On your computer, open PuTTY.
  - Host Name: `entry-dbfz.kida.thuenen.de`, Port: `22`, Connection type: `SSH`, Click the button `Open`.
  - The account name is provided in step 1. If this is the first time you login, you will asked to change the initial password.

3. **How to get code/data?**
- **From a git repository**
  - Find the URL of your repository to clone (depending on the plattform you use). Most platforms offer "Clone with SSH" and "Clone with HTTPS". Copy "Clone with HTTPS" URL.
  - Run `git clone <url>` on the HPC. If it is a private repository, you will be asked for username and password.
  - Run `ls -l` to check if the repository is cloned.
- **From your local machine**
  - See https://ticket.leipzig.dbfz.de/otobo/customer.pl?Action=CustomerFAQZoom;ItemID=616 and https://ticket.leipzig.dbfz.de/otobo/customer.pl?Action=CustomerFAQZoom;ItemID=618 to upload to `dbfz-linux-user01`.
  - When code/data is uploaded to `dbfz-linux-user01` you can use `scp` to upload it to the HPC.

4. **How to build the first image?**
- **python and pip example**
  - Copy the template file to make changes: `cp /shared/definitions/<your-institute>/pip_template.def .`
  - Add additional dependencies if needed: `vim pip_template.def` (you can use `nano` instead, or edit the file loacally)
  - Build the image: `singularity build --fakeroot /shared/<user>/image_pip.sif pip_template.def`
- **R example**
  - Copy the template file to make changes: `cp /shared/definitions/<your-institute>/r_template.def .`
  - Add additional dependencies if needed: `vim r_template.def` (you can use `nano` instead, or edit the file loacally)
  - Build the image: `singularity build --fakeroot /shared/<user>/image_r.sif r_template.def`

5. **How to test your image?**
- Create simple test file: `echo 'help("modules")' > /shared/<user>/testing.py`
- Run test file: `singularity exec /shared<user>/image.sif python3 /shared/<user>/testing.py`

6. **How to run a job with slurm?**
- In foreground: `srun singularity exec /shared/<user>/image.sif python3 /shared/<user>/testing.py`
- In background:
  - Create a file named `job.sh`. It should contain the following content:
  ```
  #!/bin/bash
  #SBATCH --output=/shared/<user>/%j.out
  #SBATCH --error=/shared/<user>/%j.err

  singularity exec -c --bind /shared/<user>:/shared/<user>,/home/<user>:/home/<user> /shared/<user>/image.sif python3 /shared/<user>/testing.py
  ```
  - Run `sbatch job.sh`. Remember the job id.
  - Run `sacct` after some seconds to see the status of your job. Identify it with the job id from the previous step.
  - Show the output from your script: `cat /shared/<user>/<jobid>.out` and `cat /shared/<user>/<jobid>.err`.


# Note
While some rules here described are enforced and mandatory, some others are not. If you have ideas to improve work on the HPC or to improve this document, please contact bernd.hassfurther@dbfz.de.
We will look forward to find a satisfying solution for everyone!
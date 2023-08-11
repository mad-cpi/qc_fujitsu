# !/bin/bash
set -e

## Matthew Dorsey
## 2023.08.10
## Script for uploading and downloading files to and from the qsim host node


## PARAMETERS
# command and flags for remote synchronization
RSYNC="rsync -Pavz"
# non-zero exit code for incorrect usage of script
declare -i NONZERO_EXITCODE=120
# boolean that determines if a path other than the defauly has been specified
declare -i PATH_BOOL=0
# default path used to upload and download files
DIR_PATH="./fujitsu"
# boolean that determines login opertation
declare -i LOGIN_BOOL=0
# boolean that determines if a directory should be uploaded to the lambda GPU instance
declare -i UPLOAD_DIR_BOOL=0
# boolean that determines if a directory should be download from the lambda GPU instance
declare -i DOWNLOAD_DIR_BOOL=0


## FUNCTIONS
# display information about script and script options / arguments
help() {

	echo -e "\n options:"
	echo " -p << ARG :: DIRECTROY PATH >> | path to directory that should be uploaded or downloaded from fujitsu"
	echo " -u                             | upload directory to qsim home directory (default directory is ${DIR_PATH})"
	echo " -d                             | download directory from qsim to current directory (default directory is ${DIR_PATH})."
	echo " -l                             | login to qsim (once other actions are completed)."
	echo " -h                             | display information about script and script options."
	echo "" 

}


# OPTIONS
# get opts
while getopts "p:udhl" option; do
	case $option in 
		p) # change path from default
			
			# parse path from option arguments
			DIR_PATH="${OPTARG}"
			;;
		u) # upload directory to qsim
			
			# boolean determining that flag was called
			declare -i UPLOAD_DIR_BOOL=1
			;;
		d) # download directory

			# boolean determing that the flag was called
			declare -i DOWNLOAD_DIR_BOOL=1
			;;
		h) # list options 
			help
			;;
		l) # login to GPU instance once other actions are completed
			
			declare -i LOGIN_BOOL=1
			;;
		\?) # default option
			
			help
			exit
			;;
	esac
done
#shift $((OPTIND-1)) 


## ARGUMENTS
# none


## SCRIPTS

# download directory from the cloud
if [[ $DOWNLOAD_DIR_BOOL -eq 1 ]]
then
	echo -e "Downloading (${DIR_PATH}) from LAMBDA INSTANCE (qsim:~).\n"
	rsync -Pavz qsim:${DIR_PATH} .
fi

# upload directory to the cloud
if [[ $UPLOAD_DIR_BOOL -eq 1 ]]
then
	echo -e "Uploading DIRECTORY (${DIR_PATH}) to LAMBDA INSTANCE (qsim:~).\n"
	rsync -Pavz ${DIR_PATH} qsim:~
fi

# login to node once other actions are completed
if [[ $LOGIN_BOOL -eq 1 ]]
then
	ssh qsim
fi

echo ""



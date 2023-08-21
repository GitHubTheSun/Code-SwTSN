# This script allows downloading ALL the raw images one by one.
# To avoid flooding our server, we ensure that images are downloaded only once (not downloading alreading existing files)
# The following variable needs to be adjusted according to your settings:
# First the variable "ALASKAv2_DATASET" specifies the version of the dataset you will download (it can be uncompressed, compressed with various quality factor and image may be of different sizes)
ALASKAv2_DATASET="ALASKA_v2_TIFF_VariousSize_GrayScale"

# More important, the variable "Image_local_path" states where the image will be downloaded; beware that this path will also be used in the script "ALASKA_conversion.sh", which generate developement of RAW files into JPEG images.
# Default setting is to put all the images into the download folder with a name that match the one used on the website ;
# To change the path where image will be dumped, simply modify the variable Image_local_path below
Image_local_path="D:\GoogleDownloads\ALASKA_v2_TIFF_VariousSize_GrayScale/"

# Last, but not least, let's specify the extension of the files (it can be either "jpg" or "tif" ; see below) ; here we guess the extension from the name of the dataset
if [ `echo $ALASKAv2_DATASET | grep -c "JPG"` -eq 1 ]
then
    filesExtension=".jpg"
elif  [ `echo $ALASKAv2_DATASET | grep -c "TIFF"` -eq 1 ]
then
    filesExtension=".tif"
else 
    echo "THE NAME OF THE DATASET SHOULD EITHER CONTAIN JPG OR TIFF"
    exit 1
fi

# Begining of the script; timestamp and creation of non-existing directory.
timeStart=$(date +%s)
if [ ! -d $Image_local_path ]; then echo "Directory $Image_local_path where images will be downloaded does not exist. It will be created" ; mkdir -p $Image_local_path; fi
if [ ! -d ./tmp ]; then mkdir -p ./tmp ; fi

# Begining of the download, since image are names 00001, 00002 ... 80000 a simple for loop allows generating all URLs
image_index=0

for (( imageIndex=1 ; imageIndex<=80005 ; imageIndex++ ));
do
    imageName=$(printf "%05d" $imageIndex)$filesExtension
    imageURL="http://alaska.utt.fr/DATASETS/$ALASKAv2_DATASET/$imageName"
    #[ -f ./tmp/$imageName ] && rm ./tmp/$imageName
    if [ ! -f  $Image_local_path$imageName ]
    then
        # ( wget -c -P ./tmp/ $imageURL && mv ./tmp/$imageName $Image_local_path$imageName ) &>> ./log_ALASKA_v2_downloads
		# wget -c -P ./tmp/ $imageURL
		wget --no-check-certificate -c -P ./tmp/ http://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_VariousSize_GrayScale/$imageName
		if [ $? -eq 0 ]; 
		then
                currentTime=$(date +%s)
				echo "Image number $imageIndex / 80005 downloaded ! Time Elapsed = $(($currentTime - timeStart)) sec. "
                exit 1
        else
                echo "download fial ,next time"
        fi
		mv ./tmp/$imageName $Image_local_path$imageName
        # This is the download command, all output is no printed out in the terminal but rather in the file "./ALASKA_log_downloads"
	# note that the -c option is for "--continue: Continue getting a partially-downloaded file." To avoid restarting partial download one can instead use  the command "[ -f ./tmp/$imageName ] && echo rm ./tmp/$imageName" above (commented) which clean/remove the file ahead
    fi
    # For progress display purpose, we print out every 10 images the number of image download with the elapsed time.
    # if [ $(expr $imageIndex % 10) -eq 0 ];
    # then
    # currentTime=$(date +%s)
    #     echo "Image number $imageIndex / 80005 downloaded ! Time Elapsed = $(($currentTime - timeStart)) sec. "
    # fi

done

# Important Note: though we provide a download script for each and every dataset, you can use this one to download all the dataset simply by changing the variable "ALASKAv2_DATASET"
# The name of those dataset is made as follows :
# ALASKA_v2_*{JPG ; TIFF}*_*SIZE*_*QF*_*{COLOR ; GrayScale}*
# where: {JPG ; TIFF} indicates the compression 
# where: *SIZE* indicates the size in the range { 256 ; 512 ; VariousSize } the later for randomly selected size
# where: *QF* indicates the JPEG compression (if not TIFF) quality factor in the range { 100 ; 95 ; 90 ; 85 ; 80 ; 75 ; QFvarious } the later for randomly selected QF
# where: *{COLOR ; GrayScale}* indicates whether images are color or grayscale
# Examples of folders containing different versions of the dataset include:
# ALASKA_v2_JPG_512_QF80_GrayScale
# ALASKA_v2_JPG_512_QF90_COLOR
# ALASKA_v2_JPG_256_QF85_COLOR
# ALASKA_v2_TIFF_VariousSize_COLOR
# ALASKA_v2_JPG_256_QFvarious_GrayScale

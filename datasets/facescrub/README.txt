==============================================================================================
FaceScrub: A Data-driven Approach To Cleaning Large Face Datasets

Authors: Hong-Wei Ng, Stefan Winkler (ADSC)

If you use or adapt any part of this dataset, please cite the following paper: 
H.-W. Ng, S. Winkler: "A Data-driven approach to cleaning large face datasets." In Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.

==============================================================================================
Description

The FaceScrub dataset contains 106863 photos of 530 celebrities, 265 whom are male, and 265 female.

The initial images that make up this dataset were obtained by searching for names of celebrities using Google Image Search.
Subsequently, we ran the Haarcascade-based face detector from OpenCV 2.4.7 on the images to obtain a set of faces for each celebrity name, with the requirement that a face must be at least 96x96 pixels (parameters: scaleFactor=1.2, minNeighbor=3, minSize = [96,96]).  We then used the method described in our paper above to identify the faces belonging to each celebrity.  Finally, any remaining mistakes are fixed by hand.

We also manually removed faces with accessories that make it impossible to recognize the person (e.g., those with completely opaque sunglasses), but faces with tinted or ordinary glasses where the eyes are still visible are kept in the dataset. Images that have undergone processing (e.g., image is darkened, converted to a sketch, warped etc.) as a form of copyright protection are also removed. However, images that have filters applied for artistic reasons are retained.

There may be multiple faces of the same person taken under the same conditions (e.g., same pose at the same event), but the images they belong to may have undergone different post-processing (e.g., resizing, compression, image filtering etc.).

Details of our method for performing the initial step of identifying the faces belonging to a celebrity can be found in our paper listed above.

Note that the number of celebrities in the FaceScrub dataset is fewer than the 695 reported in our paper, because we removed some of them for this release.
This was done for people where only a small number of face images (typically less than 100) was available, when the images were of low quality, or the face images of the person were too noisy for effective manual cleaning (e.g. when the person shares the name with another public figure).

==============================================================================================
Metadata

Due to copyright reasons we only provide the image URLs. They can be found in the files facescrub_actors.txt and facescrub_actresses.txt, for the images of male and female celebrities respectively.

Each line in a file corresponds to a face in our dataset and consists of 6 fields separated by tabs. The first line is a header giving the description of the fields. The content of the fields is the following:

1. name:
Name of the person identified in the image. Note that we only identify one person even though there may be more than one present in the image.

2. image_id: 
A number we associate with the image the face is found in. This number is unique but not necessarily in running sequence. Also, there may be multiple entries with the same image_id as there can be multiple faces of the same person in an image (e.g., where the image is an image composite). This happens even though the method described in our paper explicitly keeps at most one face per image. The additional faces are there because we manually add them back into the dataset in order not to waste them.

3. face_id: 
A number we associate with a particular face in our dataset. This number is unique for each face and is in running sequence.

4. url: 
The image file URL.

5. bbox: 
The coordinates of the bounding box for a face in the image. The format is x1,y1,x2,y2, where (x1,y1) is the coordinate of the top-left corner of the bounding box and (x2,y2) is that of the bottom-right corner, with (0,0) as the top-left corner of the image.  Assuming the image is represented as a Python Numpy array I, a face in I can be obtained as I[y1:y2, x1:x2].

6. sha256: 
The SHA-256 hash computed from the image.
After downloading an image from a given URL, the user can check if the downloaded image matches the one from our database by computing its SHA-256 hash and comparing it to the "sha256" value for the line.
On Linux systems, the SHA-256 hash can be obtained using the utility sha256sum in the shell.


The following scripts may be helpful for batch downloading of the images:
https://github.com/lightalchemist/FaceScrub/pulse
https://github.com/faceteam/facescrub

==============================================================================================
Additional information

LICENSE.txt: This is the licensing file that we recommend you read before using or sharing this dataset.

README.txt: This file.

==============================================================================================
Revision history

9-Mar-2016: Updated database to remove a number of duplicates as well as a few mislabeled and morphed faces.

28-Oct-2014: Original release

==============================================================================================

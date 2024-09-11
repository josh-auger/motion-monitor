# Container: motion-monitor
This Docker container is used for characterizing motion as measured by image registration methods that provide a series 
of alignment transform parameters to a reference.

Based on the specified input file extension (.log or .txt/.tfm), the program will either (i) comb through a single log 
file (see extract_params_from_log.py) to compile all the transform parameters that were reported during registration or 
(ii) comb through a directory and read in all transform files (see extract_params_from_transform_files.py) for the reported 
transform parameters, respectively.

### Expected outputs
Once all transform parameters have been compiled into an array list, then a series of motion measures are calculated 
(see compute_motion_measures.py). These include:
- Distribution histogram of motion transform parameters
- Displacement between adjacent acquisition instances
- Cumulative displacement over the entire scan
- Average motion per acquisition (which can be used to calculate "motion per minute" if acquisition time is known)
- Classification of image volumes as with or without motion (based on the specified motion threshold)

The motion-monitor will write all outputs and a log file (*.log) to an outputs sub-folder (./inputfilename_outputs/) 
within the parent directory that is specified in the run bash script (see Run Instructions below).

## Build Instructions
To build the motion-monitor container:
- Clone the github repository: https://github.com/josh-auger/motion-monitor
- Build the docker image with the following docker build command:
  - cd motion-monitor/
  - docker build --rm -t jauger/motion-monitor:latest -f ./Dockerfile .
  - Or execute the bash script containing the build command: sh build_docker_motion_monitor.sh

## Bash script setup
### Set input parent directory
Prior to running the container, be sure to amend the run command bash script (start_motion_monitor.sh) to specify the 
parent directory of the desired input file(s). This local directory is then shared with the container to give the 
motion-monitor access to read the files that are present.
- Open start_motion_monitor.sh in an editor
- Update the INPUT_DIR string to be the correct parent directory
- Example: 
  - INPUT_DIR="/path/to/parent/directory/of/input/files/"
  - Example log file and transform files are available at: https://drive.google.com/drive/folders/102-aBblHQNH2ILIRsIKksuJP7NvKp6BM?usp=sharing

### Set motion calculation variables
Some user-specified values are required for calculating motion and can be altered as necessary in the run command bash 
script. If no input values are specified in the bash script, the listed default values will be used.

| Variable Name      | Description                                                     | Default Value | Units  |
|--------------------|-----------------------------------------------------------------|---------------|--------|
| `head_radius`      | Spherical head radius assumption used to calculate displacement | 50            | mm     |
| `motion_threshold` | Acceptable threshold of displacement motion measure             | 0.6           | mm     |


## Run Instructions
To run the container, navigate to the motion-monitor directory and call the start_motion_monitor.sh bash script 
followed by a single input filename. The file extension of this input filename will trigger the correct data input 
method (i.e. read a log file or read a directory of transform files).
- cd motion-monitor/
- sh start_motion_monitor.sh [input filename]
- Examples:
  - For a log file: sh start_motion_monitor.sh example_rest480_slimm.log
  - For transform files: sh start_motion_monitor.sh example_rest480_sliceTransform_0002.tfm

*IMPORTANT* : One single input filename must be specified in the run command, NOT an entire filepath. For analysis of a 
directory of transform files, include one of the transform filenames as the input, typically the first file.



# References
Sui, Y., Afacan, O., Gholipour, A., & Warfield, S. K. (2020). SLIMM: Slice localization integrated MRI monitoring. 
NeuroImage, 223, 117280. https://doi.org/10.1016/J.NEUROIMAGE.2020.117280

Tisdall, M. D., Hess, A. T., Reuter, M., Meintjes, E. M., Fischl, B., & van der Kouwe, A. J. W. (2012). Volumetric 
navigators for prospective motion correction and selective reacquisition in neuroanatomical MRI. In Magnetic Resonance 
in Medicine (Vol. 68, Issue 2, pp. 389–399). John Wiley and Sons Inc. https://doi.org/10.1002/mrm.23228

Reuter, M., Tisdall, M. D., Qureshi, A., Buckner, R. L., van der Kouwe, A. J. W., & Fischl, B. (2015). Head motion 
during MRI acquisition reduces gray matter volume and thickness estimates. NeuroImage, 107, 107–115. 
https://doi.org/10.1016/j.neuroimage.2014.12.006

Pinho, A. L., Richard, H., Ponce, A. F., Eickenberg, M., Amadon, A., Dohmatob, E., Denghien, I., Torre, J. J., 
Shankar, S., Aggarwal, H., Thual, A., Chapalain, T., Ginisty, C., Becuwe-Desmidt, S., Roger, S., Lecomte, Y., 
Berland, V., Laurier, L., Joly-Testault, V., … Thirion, B. (2024). Individual Brain Charting dataset extension, 
third release for movie watching and retinotopy data. Scientific Data, 11(1), 590. 
https://doi.org/10.1038/s41597-024-03390-1

# Authorship
Unless otherwise specified, this program was created by the Computational Radiology Lab at Boston Children's Hospital, 
Boston, Massachusetts, USA.

Author(s): Auger, Joshua D.; Warfield, Simon K.\
Affiliations: Boston Children's Hospital, Harvard University\
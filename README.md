# Container: motion-monitor
This Docker container is used for characterizing motion as measured by image registration methods that provide a series 
of alignment transform parameters to a reference.

Based on the specified input file extension (.log or .txt/.tfm), the program will either (i) comb through a single log 
file (see extract_params_from_log.py) to compile all the transform parameters that were reported during registration or 
(ii) comb through a directory and read in all transform files (see extract_params_from_transform_files.py) for the reported 
transform parameters, respectively.

Once all transform parameters have been compiled into an array list, then a series of motion measures are calculated 
(see compute_motion_measures.py). These include:
- Distribution histogram of motion transform parameters
- Displacement between adjacent acquisition instances
- Cumulative displacement over the entire scan
- Average motion per minute
- Classification of image volumes as with or without motion

The motion-monitor will write all outputs and a log file (*.log) to an outputs sub-folder within the parent directory
that is specified in the run bash script (see start_motion_monitor.sh).

## Build Instructions
To build the motion-monitor container:
- Clone the github repository: https://github.com/josh-auger/motion-monitor
- Build the docker image with the following docker build command:
  - cd motion-monitor/
  - docker build --rm -t jauger/motion-monitor:latest -f ./Dockerfile .
  - Or execute the bash script containing the build command: sh build_docker_motion_monitor.sh

Some user-specified values are hard-coded into the motion measure analysis (see compute_motion_measures.py, beginning 
with line 384). These values can be altered as necessary.

| Variable Name    | Description                                                                                 | Default Value | Units  |
|------------------|---------------------------------------------------------------------------------------------|---------------|--------|
| `radius`         | Spherical head radius assumption used to calculate displacement                             | 50            | mm     |
| `threshold_value`| Displacement threshold for acceptable motion                                                | 0.75          | mm     |
| `acquisition_time`| Time between each instance of image acquisition used to calculate average motion per minute | 4.2           | sec    |

If any user-specified values are altered in the source code, be sure to re-build the motion-monitor docker container 
following the prior steps.

## Run Instructions
Prior to running the container, be sure to amend the run command bash script (start_motion_monitor.sh) to specify the 
parent directory of the desired input file(s).
- Open start_motion_monitor.sh in an editor
- Update the INPUT_DIR string to be the correct directory
- Example: 
  - INPUT_DIR="/path/to/input/file/"
  - Example log file and transform files are available at: https://drive.google.com/drive/folders/102-aBblHQNH2ILIRsIKksuJP7NvKp6BM?usp=sharing

To run the container, navigate to the motion-monitor directory and execute the run command bash script with a 
specified input filename. The file extension will trigger the correct data input method (log file or directory
of transform files).
- cd motion-monitor/
- sh start_motion_monitor.sh [input filename]
- Examples:
  - For a log file: sh start_motion_monitor.sh example_rest480_slimm.log
  - For transform files: sh start_motion_monitor.sh example_rest480_sliceTransform_0002.tfm

- NOTE: A single input filename must be specified in the run command. For analysis of a directory of transform files,
simply include one of the transform filenames as the input, typically the first file.



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
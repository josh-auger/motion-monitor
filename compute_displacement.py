#!/usr/bin/env python3

import SimpleITK as sitk
import numpy as np
import argparse

#
# This program reads in two affine transforms.
# Each transform maps from a reference volume to a target volume.
# The second transform maps from a reference volume to a target volume.
# The first transform is inverted and composed with the second transform to 
#   create a transform that maps from the 
#       first target volume to the second target volume.
# The composed transform is written to a file.
#

# We believe that angles are always in radians in SimpleITK.

def compute_displacement(transform1, transform2, outputfile=None):

    # Review the definition of transformations here:
    # https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/22_Transforms.html
    # https://simpleitk.org/SPIE2019_COURSE/01_spatial_transformations.html
    #
    # T(x)=A(x−c)+t+c
    # A is a 3x3 matrix.
    # t is a 3x1 vector.
    # c is the center of rotation, a point represented by a 3x1 vector.

    A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
    c0 = np.asarray(transform2.GetCenter())
    t0 = np.asarray(transform2.GetTranslation())

    A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
    c1 = np.asarray(transform1.GetInverse().GetCenter())
    t1 = np.asarray(transform1.GetInverse().GetTranslation())

    # Create a single transform manually.
    # this is a recipe for compositing any two global transformations
    # into an affine transformation, T_0(T_1(x)):
    # A = A=A0*A1
    # c = c1
    # t = A0*[t1+c1-c0] + t0+c0-c1
    #
    # T_0(A0,c0,t0)(x) = A0(x-c0) + c0 + t0
    # T_1(A1,c1,t1)(x) = A1(x-c1) + c1 + t1
    # T_0(T_1(x)) = A0( A1(x-c1) + c1 + t1 - c0 ) + c0 + t0
    #             = A0A1(x-c1) + A0(c1 + t1 - c0) + c0 + t0
    #             = A0A1(x-c1) + c1 - c1 + A0(c1 + t1 - c0) + c0 + t0
    #             = A0A1(x-c1) + c1 + t
    #               where t = A0(t1 + c1 - c0) + t0 + c0 - c1
    #

    combined_mat = np.dot(A0,A1)
    combined_center = c1
    combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1
    combined_affine = sitk.AffineTransform(combined_mat.flatten(), combined_translation, combined_center)

    # print('\nCombined affine transform : ')
    # print(combined_affine)
    # print(combined_affine.GetParameters())
    # print(combined_affine.GetFixedParameters())

    # Save composed transform to outputfile
    if outputfile:
        sitk.WriteTransform(combined_affine, outputfile)

    versorrigid3d = sitk.VersorRigid3DTransform()
    versorrigid3d.SetCenter(combined_center)
    versorrigid3d.SetTranslation(combined_translation)
    versorrigid3d.SetMatrix(combined_mat.flatten())
    # print('\n')
    # print(versorrigid3d)
    # print("Composed parameters (VersorRigid3D) : ", np.asarray(versorrigid3d.GetParameters()))

    # First three parameters are rotation angles in radians.
    # Second three parameters are translations.

    euler3d = sitk.Euler3DTransform()
    euler3d.SetCenter(combined_center)
    euler3d.SetTranslation(combined_translation)
    euler3d.SetMatrix(combined_mat.flatten())
    # print('\n')
    # print(euler3d)
    # print(euler3d.GetParameters())

    # Compute the displacement:
    radius = 50
    params = np.asarray( euler3d.GetParameters() )
    print("Composed parameters (Euler3D) : ", params)

    # # Original method: l1 norm (over-estimates translations)
    # displacement = abs(params[0]*radius) + abs(params[1]*radius) + \
    #     abs(params[2]*radius) + abs(params[3]) + abs(params[4]) + abs(params[5])

    # # Alternative method: l2 norm of magnitude of rotation and translations (over-estimates rotation contributions)
    # displacement = (radius * np.sqrt((params[0]**2) + (params[1]**2) + (params[2]**2))) + \
    #                 np.sqrt((params[3] ** 2) + (params[4] ** 2) + (params[5] ** 2))

    # Tisdall et al. 2012 exact equation
    theta = np.abs(np.arccos(0.5 * (-1 + np.cos(params[0]) * np.cos(params[1]) + \
                                    np.cos(params[0]) * np.cos(params[2]) + \
                                    np.cos(params[1]) * np.cos(params[2]) + \
                                    np.sin(params[0]) * np.sin(params[1]) * np.sin(params[2]))))
    drot = radius * np.sqrt((1 - np.cos(theta)) ** 2 + np.sin(theta) ** 2)
    dtrans = np.linalg.norm(params[3:])
    displacement = drot + dtrans

    print("Displacement : ", displacement)

    return displacement

if __name__ == "__main__":
    # Example usage
    transform1 = sitk.ReadTransform("/home/jauger/Radiology_Research/SLIMM_data/20240321_restingstate_480vols/20240701_slow_local_server_registration_recursive_transforms/rest480_BOBYQA_recursivetransform_test1/sliceTransform_0002.tfm")
    transform2 = sitk.ReadTransform("/home/jauger/Radiology_Research/SLIMM_data/20240321_restingstate_480vols/20240701_slow_local_server_registration_recursive_transforms/rest480_BOBYQA_recursivetransform_test1/sliceTransform_0003.tfm")
    displacement_result = compute_displacement(transform1, transform2)
    # print("Displacement:", displacement_result)

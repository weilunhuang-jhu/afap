# As Focus As Possible (AFAP)
This code-base implements the method presented in the paper [A Novel Method to Improve Quality Surface Coverage in Multi-View Capture](https://arxiv.org/abs/2407.15883).
Given the 3D mesh of the target object and camera poses in a scan, we would like to derive a focus distance for each camera that optimizes the quality of the covered surface area captured in the images.

![Proposed Method](https://i.imgur.com/Rq3pynG.png)

## Introduction:
The depth of field of a camera is a limiting factor for applications that require taking images at a short subject-to-camera distance or using a large focal length, such as total body photography, archaeology, and other close-range photogrammetry applications. Furthermore, in multi-view capture, where the target is larger than the camera's field of view, an efficient way to optimize surface coverage captured with quality remains a challenge. Given the 3D mesh of the target object and camera poses, we propose a novel method to derive a focus distance for each camera that optimizes the quality of the covered surface area. We first design an Expectation-Minimization (EM) algorithm to assign points on the mesh uniquely to cameras and then solve for a focus distance for each camera given the associated point set. We further improve the quality surface coverage by proposing a $k$-view algorithm that solves for the points assignment and focus distances by considering multiple views simultaneously. We demonstrate the effectiveness of the proposed method under various simulations for total body photography. Please refer to the [paper](https://arxiv.org/abs/2407.15883) for details.

## Installation:

* conda: conda-forge channel
* python: 3.8 for Windows, 3.9 for Linux
* Essential: vtk, opencv, trimesh, rtree, igl, pyembree, open3d
* Vis: pyvista, pyvistaqt, matplotlib, pyqt5 
* Test: networkx (pip)

### Installation command (Windows)
```
conda create --name afap -c conda-forge python=3.8 pyvista pyvistaqt opencv trimesh rtree igl matplotlib
conda activate afap
pip install open3d
pip install pyembree==0.2.11
```

### Installation command (Linux)
```
conda create --name afap -c conda-forge python=3.9 pyvista pyvistaqt opencv trimesh rtree igl matplotlib embree=2.17.7 pyembree
conda activate afap
pip install open3d
```

## Data

The textured 3D human mesh model can be downloaded: [3DBodyTex.v1](https://cvi2.uni.lu/datasets/).

## Preprocess data

In [script/](https://github.com/weilunhuang-jhu/afap/tree/main/script)

- Transform data: (convert to mm scale)
```
python transform_3dbodytex.py -i path_to_3dbodytex_data
```
## Usage

### Run the proposed algorithm

- EM algorithm:

```
python EM.py -i ../data/3dbodytex/005 -c camera_poses_cylinder_7_24_266_750.pkl -p ../params/test.yml -s output_em
```

- $k$-View algorithm:

```
python KViews.py -i ../data/3dbodytex/005 -c camera_poses_cylinder_7_24_266_750.pkl -p ../params/test.yml -s output_kview 
```

- $k$-View algorithm initialized with EM results
```
python KViews.py -i ../data/3dbodytex/005 -c camera_poses_cylinder_7_24_266_750.pkl -p ../params/test.yml -s output_kview -ini ../data/005/output_em/output_iter_6.pkl 
```


### Visualization

![Comparison 3D](https://i.imgur.com/wdgUyQm.png)

![Comparison Image](https://i.imgur.com/Oz5m8jQ.png)
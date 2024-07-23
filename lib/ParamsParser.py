import yaml

def load_params(path):
    with open(path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def get_camera_params(params):
    camera_params = {'aspect_ratio' : params['camera']['aspect_ratio'],\
                    'focal_length' : params['camera']['focal_length'],\
                    'hyperfocal_distance' : params['camera']['hyperfocal_distance']
                     }
    return camera_params

def get_loss_params(params):
    loss_params = {'mode' : params['loss']['mode'],\
                    'weights' : params['loss']['weights'],\
                    'projection_area_loss_type' : params['loss']['projection_area_loss_type'],\
                    'off_optical_axis_loss_type' : params['loss']['off_optical_axis_loss_type'],\
                    'focal_loss_type' : params['loss']['focal_loss_type'],\
                    'projection_area_threshold' : params['loss']['projection_area_threshold'],\
                    'off_optical_axis_threshold_distance' : params['loss']['off_optical_axis_threshold_distance'],\
                    'num_focus_intervals' : params['loss']['num_focus_intervals']
                     }
    assert loss_params['num_focus_intervals'] >= 4, "num_focus_intervals should be at least 4"
    assert loss_params['num_focus_intervals'] % 2 == 0, "num_focus_intervals should be even"

    return loss_params
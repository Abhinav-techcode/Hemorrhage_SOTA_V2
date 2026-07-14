from monai.transforms import RandCropByPosNegLabeld
class SinglePosNegCropd:
    def __init__(self, keys, label_key, spatial_size, pos, neg, image_key, image_threshold):
        self.cropper = RandCropByPosNegLabeld(
            keys=keys, label_key=label_key, spatial_size=spatial_size,
            pos=pos, neg=neg, num_samples=1, image_key=image_key, image_threshold=image_threshold
        )
    def __call__(self, data):
        out = self.cropper(data)
        return out[0]

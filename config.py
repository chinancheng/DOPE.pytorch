class Config(object):

    stride = 8
    crop_size = 256
    sigma = 5.0
    test_sigma = 0.1
    thresh_map = 0.1
    thresh_points = 0.5
    thresh_angle = 0.1
    grid = crop_size / stride
    vec_width = 3.0
    output_stage = 6
    threshold = 0.1
    @classmethod
    def describe(cls):
        text = 'Config:\n'
        attrs = [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith('__')]
        for attr in attrs:
            text += '\t{:s} = {:s}\n'.format(attr, str(getattr(cls, attr)))
        return text


if __name__ == '__main__':
    def main():
        print(Config.describe())
    main()

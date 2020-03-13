import os

# cslab cluster
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/u/zianwang/disk/dataset/PASCAL/'     # folder that contains VOCdevkit/.

        elif database == 'sbd':
            return '/u/zianwang/disk/dataset/SBD/'        # folder with benchmark_RELEASE/

        elif database == 'davis2016':
            return '/u/zianwang/disk/dataset/DAVIS/'      # folder with Annotations/, ImageSets/, JPEGImages/, ...

        elif database == 'cityscapes-processed':
            return '/u/zianwang/disk/dataset/polyrnn-pp-pytorch/data/cityscapes_processed'

        elif database == '360vSOD':
            return (os.getcwd() + '/dataset/' + '360vsod/')

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return os.getcwd() + '/models/'
       # return 'models/'

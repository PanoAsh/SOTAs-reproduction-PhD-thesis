import os

train_pair_path = os.path.join('./data/360ISOD', 'train_pair.lst')
test_path = os.path.join('./data/360ISOD-TE', 'test.lst')
train_path = os.path.join('./train.lst')
edge_path = os.path.join('./edge.lst')
png_path = os.getcwd() + '/data/360ISOD-TE/img_test'
jpg_path = os.getcwd() + '/data/360ISOD-TE/img_test'

class lst_file():
    def __init__(self):
        self.pathI = os.getcwd() + '/data/360ISOD/360ISOD-Image'

    def lst_generation(self):
        imglist = os.listdir(self.pathI)
        imglist.sort(key=lambda x: x[:-4])

        f = open(train_pair_path, 'w')

        for item in imglist:
            line = '360ISOD-Image' + '/' + item + ' ' + '360ISOD-Mask' + '/' + item + '\n'
            f.write(line)
        f.close()

    def lst_test(self):
        imglist = os.listdir(jpg_path)
        imglist.sort(key=lambda x: x[:-4])

        f = open(train_path, 'w')

        count = 1
        for item in imglist:
            line = item[:-3] + 'jpg' + '\n'
            f.write(line)
            print(" {} images processed".format(count))
            count += 1
        f.close()

    def png2jpg(self):
        filelist = os.listdir(png_path)

        count = 1
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(png_path), item)
                dst = os.path.join(os.path.abspath(jpg_path), item[:-4] + '.jpg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                except:
                    continue
            print(" {} images processed".format(count))
            count += 1


if __name__ == '__main__':
    lf = lst_file()

   # lf.lst_generation()
    #lf.png2jpg()
    lf.lst_test()

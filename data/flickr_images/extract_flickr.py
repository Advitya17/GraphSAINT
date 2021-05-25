import numpy as np
import pandas as pd
import os, glob

feats = np.load('../flickr/feats.npy')
feat_lookup = np.loadtxt('../BoW_int.dat')

urls = pd.read_fwf('../NUS-WIDE-urls.txt')

left_images = open('left_images.txt', 'w')

curr, prob = len(glob.glob('*')), 0
for img_f in feats:
    for idx, f in enumerate(feat_lookup):
        if np.array_equal(img_f, f):
            # web scrape
            #link = urls.iloc[idx, 0].split()[2]
            #os.system('wget ' + link)
            
            # local move
            link = urls.iloc[idx, 0].split()[0]
            os.system('mv ../' + '/'.join(link.split('/')[2:]) + ' .')

            num_files = len(glob.glob('*'))
            if num_files == curr + 1:
                curr += 1
            elif num_files == curr:
                prob += 1
                left_images.write(link + '\n')
            else:
                raise
            break
left_images.close()
print('Number of images not found: ', str(prob))

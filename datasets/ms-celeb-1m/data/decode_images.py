import multiprocessing
import base64
import io
from PIL import Image
import os
import hashlib


def decode_image((url, face_data)):
    results_path = './images_test'
    sha_res = hashlib.sha1(url).hexdigest() + '.jpg'
    fname = os.path.join(results_path, sha_res)
    
    decoded = base64.b64decode(face_data)
    img = Image.open(io.BytesIO(decoded))
    img.save(fname)

    info_path = './'
    with open(os.path.join(info_path,'test_data_info.txt'), 'a') as fd:
                fd.write('%s\n' % (sha_res))
        
if __name__=='__main__':
    data_file = 'MsCelebV1-Faces-Cropped-DevSet2.tsv'
    num_to_decode = 500
    
    print('Reading in file...')
    tasks = []
    with open(data_file, 'r') as f:
        for i in range(0, num_to_decode):
            line = f.readline()
            tokens = line.split('\t')
            tasks.append((tokens[2], tokens[5]))
            
    print('Done.')
    print('Decoding...')
    
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
    pool.map(decode_image, tasks)
    pool.close()
    pool.join()       
    
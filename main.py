import processing
 
rootdir = './csidata/1_distortion_objects/1'
cats = processing.read.categorize(processing.read.listdirs(rootdir), ['bottle', 'empty'])
print(cats)
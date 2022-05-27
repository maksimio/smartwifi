import processing
 
rootdir = './csidata/1_metal_objects'
cats = processing.read.categorize(processing.read.listdirs(rootdir), ['bottle', 'empty'])
print(cats)
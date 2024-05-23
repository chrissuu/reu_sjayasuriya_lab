from PIL import Image
import numpy

header = "../data/SVSS_Sample_Deep_194356/"

im_x = Image.open(header + 'mip_x.tif')
im_y = Image.open(header + 'mip_y.tif')
im_z = Image.open(header + 'mip_z.tif')

# im_x.show()

im_x_array = numpy.array(im_x)
print(im_x_array)

for i in enumerate(im_x_array):
    raise_flag = False
    for elt in i[1]:
        if elt != 0:
            raise_flag = True
        if raise_flag:
            print(elt)
            raise_flag = False
    



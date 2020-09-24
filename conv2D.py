import torch
import torch.nn.functional as F
import numpy as np
import unittest


np.random.seed(42)

def conv2D(kernel, img):
    Co = kernel.shape[0]
    Kh = kernel.shape[2]
    Kw = kernel.shape[3]
    B = img.shape[0]
    Ci = img.shape[1]
    Ih = img.shape[2]
    Iw = img.shape[3]
    out = np.zeros((B,Co,Ih-Kh+1,Iw-Kw+1), dtype='float32')

    dot_p = 0
    for b in range(B):
        for k in range(Co):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    for x in range(Kh):
                        for y in range(Kw):
                            for z in range(Ci):
                                dot_p += kernel[k,z,x,y] * img[b][z][i+x][j+y]
                    out[b,k,i,j] = dot_p
                    dot_p = 0

    return torch.from_numpy(out)

class Test(unittest.TestCase):

    def test_1(self):
        img = torch.rand((1,1,4,4))
        kernel = torch.rand((1,1,2,2))
        out_1 = conv2D(kernel ,img)
        out_2 = F.conv2d(img, kernel)
        res = torch.allclose(out_1,out_2)
        self.assertEqual(res, True)

    def test_2(self):
        img = torch.rand((1,1,32,32))
        kernel = torch.rand((1,1,3,3))
        out_1 = conv2D(kernel ,img)
        out_2 = F.conv2d(img, kernel)
        res = torch.allclose(out_1,out_2)
        self.assertEqual(res, True)

    def test_3(self):
        img = torch.rand((1,3,4,4))
        kernel = torch.rand((3,3,2,2))
        out_1 = conv2D(kernel ,img)
        out_2 = F.conv2d(img, kernel)
        res = torch.allclose(out_1,out_2)
        self.assertEqual(res, True)

    def test_4(self):
        img = torch.rand((1,3,32,32))
        kernel = torch.rand((9,3,3,3))
        out_1 = conv2D(kernel ,img)
        out_2 = F.conv2d(img, kernel)
        res = torch.allclose(out_1,out_2)
        self.assertEqual(res, True)

    def test_5(self):
        img = torch.rand((5,3,4,4))
        kernel = torch.rand((3,3,2,2))
        out_1 = conv2D(kernel ,img)
        out_2 = F.conv2d(img, kernel)
        res = torch.allclose(out_1,out_2)
        self.assertEqual(res, True)

    def test_6(self):
        img = torch.rand((3,3,32,32))
        kernel = torch.rand((9,3,3,3))
        out_1 = conv2D(kernel ,img)
        out_2 = F.conv2d(img, kernel)
        res = torch.allclose(out_1,out_2)
        self.assertEqual(res, True)

if __name__ == "__main__":
    unittest.main()
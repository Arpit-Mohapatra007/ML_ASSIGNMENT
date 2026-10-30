import cv2, numpy as np, gradio as gr
from predict import predict as pp_

def f(p):
    lab, pr = pp_(p)
    return lab if pr is None else f'{lab} (p={pr:.2f})'

if __name__ == '__main__':
    cv2.imwrite('blank.png', np.full((512, 512, 3), 128, np.uint8))
    cv2.imwrite('dog_grey.png', cv2.imread('dog.jpg', 0))
    for n in ['blank.png', 'dog.jpg', 'dog_grey.png']:
        print(n, pp_(n))
    gr.Interface(f, gr.Image(type='filepath'), 'text').launch()

import time
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
import io

from model.network import deeplabv3plus_resnet50 as S2M
from model.aggregate import aggregate_wbg_channel as aggregate
from dataset.range_transform import im_normalization
from util.tensor_util import pad_divide_by

class InteractiveManager:
    def __init__(self, model, image, mask):
        self.model = model

        self.image = im_normalization(TF.to_tensor(image)).unsqueeze(0)
        self.mask = TF.to_tensor(mask).unsqueeze(0)

        h, w = self.image.shape[-2:]
        self.image, self.pad = pad_divide_by(self.image, 16)
        self.mask, _ = pad_divide_by(self.mask, 16)
        self.last_mask = None

        # Positive and negative scribbles
        self.p_srb = np.zeros((h, w), dtype=np.uint8)
        self.n_srb = np.zeros((h, w), dtype=np.uint8)

        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.positive_mode = True
        self.need_update = True

    def run_s2m(self):
        # Convert scribbles to tensors
        Rsp = torch.from_numpy(self.p_srb).unsqueeze(0).unsqueeze(0).float()
        Rsn = torch.from_numpy(self.n_srb).unsqueeze(0).unsqueeze(0).float()
        Rs = torch.cat([Rsp, Rsn], 1)
        Rs, _ = pad_divide_by(Rs, 16)

        # Use the network to do stuff
        inputs = torch.cat([self.image, self.mask, Rs], 1)
        _, mask = aggregate(torch.sigmoid(self.model(inputs)))

        # We don't overwrite current mask until commit
        self.last_mask = mask
        np_mask = (mask.detach().cpu().numpy()[0,0] * 255).astype(np.uint8)

        if self.pad[2]+self.pad[3] > 0:
            np_mask = np_mask[self.pad[2]:-self.pad[3],:]
        if self.pad[0]+self.pad[1] > 0:
            np_mask = np_mask[:,self.pad[0]:-self.pad[1]]

        return np_mask

    def commit(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        if self.last_mask is not None:
            self.mask = self.last_mask

    def clean_up(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        self.mask.zero_()
        self.last_mask = None

def comp_image(image, mask, p_srb, n_srb):
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:,:,2] = 1
    if len(mask.shape) == 2:
        mask = mask[:,:,None]
    comp = (image*0.5 + color_mask*mask*0.5).astype(np.uint8)
    comp[p_srb>0.5, :] = np.array([0, 255, 0], dtype=np.uint8)
    comp[n_srb>0.5, :] = np.array([255, 0, 0], dtype=np.uint8)
    return comp

@st.cache_resource
def load_model():
    net = S2M()
    net.load_state_dict(torch.load('saves/s2m.pth', map_location='cpu'))
    net = net.eval()
    torch.set_grad_enabled(False)
    return net

def main():
    container = st.container()

    # network stuff
    net = load_model()
    bg_image_file = container.file_uploader("首帧图片", type=['jpg', 'png'], label_visibility="hidden")
    bg_image = Image.open(bg_image_file).convert('RGB') if bg_image_file is not None else None
    col1, col2 = container.columns(2)

    if bg_image:
        shape = bg_image.size
        if shape[0] > 1800:
            w = shape[0] // 4
            h = shape[1] // 4
        elif shape[0] > 1000:
            w = shape[0] // 2
            h = shape[1] // 2
        else:
            w = shape[0]
            h = shape[1]

        img = bg_image.resize((w, h), Image.ANTIALIAS)
        mask = np.zeros((h, w), dtype=np.uint8)

        manager = InteractiveManager(net, img, mask)

        # def mode_change():
        #     manager.commit()

        with col1:
            col11, col12, col13 = col1.columns(3)
            # Specify canvas parameters in application
            drawing_mode = col11.selectbox(
                "Drawing tool:",
                ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
            )
            stroke_width = col12.slider("Stroke width: ", 1, 25, 3)

            if drawing_mode == 'point':
                point_display_radius = col13.slider("Point display radius: ", 1, 25, 3)
            canvas_result = None


            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 1)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color="rgba(1, 123, 56, 0.6)",
                background_color="#eee",
                background_image=img,
                update_streamlit=True,
                width=w,
                height=h,
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                display_toolbar=True,
                key="full_app",
            )
        
        with col2:
            col21, col22 = col2.columns(2)
            # modes = col21.radio("Choose draw mode.", ('Positive','Negative'), on_change=mode_change)
            show = col21.radio("Choose display mode.", ('Mask', 'Composive'))
            # isPos = True if modes == 'Positive' else False
            showMask = True if show == 'Mask' else False

            # Do something interesting with the image data and paths
            if canvas_result is not None and canvas_result.image_data is not None:
                # if isPos:
                manager.p_srb = canvas_result.image_data[:,:,0]
                # else:
                #    manager.n_srb = canvas_result.image_data[:,:,0]
                pre = time.time()
                np_mask = manager.run_s2m()
                out = cv2.resize(np_mask, (shape[0],shape[1]))
                buf = io.BytesIO()
                Image.fromarray(out).save(buf, format='png')
                dur = time.time() - pre
                col2.text(dur)
                if showMask:
                    st.image(out, caption="mask", width=w)
                else:
                    display = comp_image(np.asarray(img), np_mask, manager.p_srb, manager.n_srb)
                    st.image(display, caption="合成图片")
                col22.download_button(
                    label="Download the mask",
                    data=buf,
                    file_name='00000.png',
                    mime='image/png',
                )


if __name__ == "__main__":
    st.set_page_config(
        page_title="交互生成首帧mask", page_icon=":pencil2:",
        layout="wide"
    )
    st.title("首帧mask生成器")
    main()

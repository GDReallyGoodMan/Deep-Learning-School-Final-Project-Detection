from ultralytics import YOLO
import torch
import onnxruntime as ort
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import tempfile
import time
import zipfile
import io
import streamlit as st



if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []

def create_zip_from_memory(image_data_list):
    """Создает ZIP архив из списка изображений в памяти"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, image_bytes in image_data_list:
            zip_file.writestr(filename, image_bytes)
    zip_buffer.seek(0)
    return zip_buffer





@st.cache_resource
def load_accurate_model():
    session_accurate = YOLO('ml/M_detect_vehicle.pt',task="detect")
    return session_accurate 

@st.cache_resource
def load_fast_model():
    session_fast = YOLO('ml/S_detect_vehicle.pt',task="detect")
    return session_fast 



session_fast = load_fast_model()





left_col, cent_col, last_col = st.columns([1,4,1])

with cent_col: 
    model_choice = st.radio(
        "Выберите модель:",
        ["🚀 Быстрая", 
        "🎯 Точная"],
        horizontal=True
    )


with cent_col: 
    st.space('large')


with cent_col: 
    uploaded_imgs = st.file_uploader(
        "Загрузите фото машин", 
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=True, 
        key="file_uploader"
    )



with cent_col: 
    st.space('large')






with cent_col: 
    if st.button("🔍 Найти машины", type="primary", use_container_width=True):
        if (len(uploaded_imgs) == 0): st.warning("⚠️ Вы не загрузили ни одного изображения")
        else:
            st.session_state.processed_images.clear()
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            imgs_len = len(uploaded_imgs)

            model = session_fast
            if (model_choice == "🎯 Точная"):
                    model = load_accurate_model()

            for i, img in enumerate(uploaded_imgs):
                status_text.text(f"Обрабатываю изображение {i+1} из {imgs_len}")
                progress_bar.progress((i + 1) / imgs_len)

                img = Image.open(img)
                img_np = np.array(img)
                res = model(img_np)
                im_array = res[0].plot()

                st.image(im_array)

                im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                success, encoded_image = cv2.imencode('.jpg', im_array_rgb)  
                if success:
                    image_bytes = encoded_image.tobytes()
                    filename = f"processed_{i+1}_{uploaded_imgs[i].name}"
                    st.session_state.processed_images.append((filename, image_bytes))


                if (model_choice == '🚀 Быстрая'):
                    time.sleep(0.1) 
                else:
                    time.sleep(0.3)


                
            st.success("Готово!")

            status_text.empty()
            progress_bar.empty()

            if st.session_state.processed_images:
                st.markdown("---")
                zip_buffer = create_zip_from_memory(st.session_state.processed_images)
                
                st.download_button(
                    label=f"📥 Скачать все обработанные изображения ({len(st.session_state.processed_images)} шт.)",
                    data=zip_buffer,
                    file_name="processed_cars.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="download_zip"
                )
                
                with st.expander("📊 Информация об архиве"):
                    st.write(f"**Количество файлов:** {len(st.session_state.processed_images)}")
                    st.write(f"**Размер архива:** {len(zip_buffer.getvalue()) // 1024} KB")
                    st.write("**Список файлов в архиве:**")
                    for filename, _ in st.session_state.processed_images:
                        st.write(f"• {filename}")
